from openai import OpenAI, AsyncOpenAI
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Union, Any
import asyncio
from pydantic import BaseModel
import logging
import litellm
from pathlib import Path
from google import genai
import anthropic
import base64
import httpx
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parsable = ["gpt-4o", "o1"]
openai_models = ["gpt-4o", "o1", "gpt-4o-mini", "o1-mini", "gpt-4.1", "o4-mini"]
deepseek_models = ["deepseek-reasoner", "deepseek-chat"]
anthropic_models = ["claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022", "claude-3-7-sonnet-20250219", "claude-4-sonnet-20250514"]
gemini_models = ["gemini-2.0-flash-thinking-01-21", "gemini-2.0-pro-exp-02-05"]
parsing_model = "gpt-4o-mini"

##################### Batch processing #########################
class LLMProcessor:
    def __init__(self, max_concurrent: int = 25, model: str = "gpt-4o", max_tokens: int = 4096):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.model = model
        self.max_tokens = max_tokens
        self.parsable = self.model in parsable
        
    def _encode_image_to_base64(self, image_path_or_url: str) -> tuple:
        """
        Encode an image to base64 from a file path or URL.
        Returns a tuple of (base64_data, media_type)
        """
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                # It's a URL
                response = httpx.get(image_path_or_url)
                response.raise_for_status()
                image_data = response.content
                # Try to determine media type from content
                try:
                    img = Image.open(BytesIO(image_data))
                    media_type = f"image/{img.format.lower()}"
                except Exception:
                    # Default to jpeg if we can't determine
                    media_type = "image/jpeg"
            else:
                # It's a file path
                with open(image_path_or_url, "rb") as image_file:
                    image_data = image_file.read()
                # Try to determine media type from file extension
                ext = Path(image_path_or_url).suffix.lower().lstrip('.')
                if ext in ['jpg', 'jpeg']:
                    media_type = "image/jpeg"
                elif ext in ['png']:
                    media_type = "image/png"
                elif ext in ['gif']:
                    media_type = "image/gif"
                elif ext in ['webp']:
                    media_type = "image/webp"
                else:
                    # Try to determine from content
                    try:
                        img = Image.open(BytesIO(image_data))
                        media_type = f"image/{img.format.lower()}"
                    except Exception:
                        # Default to jpeg if we can't determine
                        media_type = "image/jpeg"
            
            # Encode to base64
            base64_data = base64.b64encode(image_data).decode("utf-8")
            return base64_data, media_type
        except Exception as e:
            logging.error(f"Error encoding image: {str(e)}")
            raise

    def _format_message_with_images(self, content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Format a message with images for Claude models.
        Content can be a string or a list of content blocks.
        """
        return [{"role": "user", "content": content}]
        

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_api_sync(self, messages: List[Dict], parse=None, get_response =False) -> str:
        """
        Synchronous version of the call. 
        If 'parse' is supplied and the model is in 'parsable', 
        we do one-pass structured parse. Otherwise, 
        we do a two-pass approach (litellm first, then OpenAI parse).
        """
        max_retries = 3
        attempt = 0
        
        while attempt < max_retries:
            try:
                # -----------------------------------------------------------
                if self.model in openai_models:
                    if self.model == "deepseek-reasoner":
                        client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
                    else:
                        client = OpenAI()

                # 1) Single-pass structured parse if the model supports it:
                if parse and self.parsable:
                    # Use openai's parse directly if your model supports it
                    # (Assuming you have some custom openai client with a .beta endpoint)

                    response = client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        response_format=parse
                    )
                    return response.choices[0].message.parsed

                # -----------------------------------------------------------
                # 2) Two-pass approach: litellm for content, then openai parse:
                elif parse and not self.parsable:
                    # First pass with litellm
                    if self.model in openai_models:
                        litellm_response = client.chat.completions.create(
                            model=self.model,
                            messages=messages
                        )
                        unparsed_response = litellm_response.choices[0].message.content.strip()
                    elif self.model in anthropic_models:
                        client = anthropic.Anthropic(
                            api_key=os.environ.get("ANTHROPIC_API_KEY"),
                        )
                        # Check if the message contains images
                        if isinstance(messages[0]["content"], list):
                            response = client.messages.create(
                                model=self.model,
                                max_tokens=self.max_tokens,
                                messages=[{
                                    "role": messages[0]["role"],
                                    "content": messages[0]["content"]
                                }]
                            )
                        else:
                            response = client.messages.create(
                                model=self.model,
                                max_tokens=self.max_tokens,
                                messages=messages
                            )
                        unparsed_response = response.content[0].text.strip()
                    else:
                        litellm_response = litellm.completion(
                            model=self.model,
                            messages=messages
                        )
                        unparsed_response = litellm_response["choices"][0]["message"]["content"].strip()

                    # Second pass with openai parse
                    client = OpenAI()
                    response = client.beta.chat.completions.parse(
                        model=parsing_model,
                        messages=[{
                            "role": "user",
                            "content": f"Given the following data, format it with the given response format: {unparsed_response}"
                        }],
                        response_format=parse
                    )
                    return response.choices[0].message.parsed

                # -----------------------------------------------------------
                # 3) If no parse needed, just do one pass with litellm:
                else:
                    if self.model in openai_models:
                        if get_response:
                            response = client.responses.create(
                                model=self.model,
                                input=messages
                            )
                            return response.output_text.strip()
                        else:
                            response = client.chat.completions.create(
                                model=self.model,
                                messages=messages
                            )
                            return response.choices[0].message.content.strip()
                    elif self.model in anthropic_models:
                        client = anthropic.Anthropic(
                            api_key=os.environ.get("ANTHROPIC_API_KEY"),
                        )
                        # Check if the message contains images
                        if isinstance(messages[0]["content"], list):
                            response = client.messages.create(
                                model=self.model,
                                max_tokens=self.max_tokens,
                                messages=[{
                                    "role": messages[0]["role"],
                                    "content": messages[0]["content"]
                                }]
                            )
                        else:
                            response = client.messages.create(
                                model=self.model,
                                max_tokens=self.max_tokens,
                                messages=messages
                            )
                        return response.content[0].text.strip()
                    else:
                        litellm_response = litellm.completion(
                            model=self.model,
                            messages=messages
                        )
                        return litellm_response["choices"][0]["message"]["content"].strip()

            except Exception as e:
                attempt += 1
                # print(messages)
                if attempt < max_retries:
                    logging.warning(
                        f"Attempt {attempt} failed with error. "
                        f"Waiting 10 seconds before retry...\nError: {str(e)}"
                    )
                    time.sleep(10)
                else:
                    logging.error(f"Failed after {max_retries} attempts. Error: {str(e)}")
                    return None

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _call_api_async(self, messages: List[Dict], parse=None, get_response=False) -> str:
        """
        Asynchronous version of the call.
        Same logic: single-pass if model is parsable, else two-pass
        (first litellm, then openai parse).
        """
        async with self.semaphore:
            max_retries = 10
            attempt = 0
            while attempt < max_retries:
                try:
                    # 1) Single-pass parse if model supports it
                    if parse and self.parsable:
                        client = AsyncOpenAI()
                        response = await client.beta.chat.completions.parse(
                            model=self.model,
                            messages=messages,
                            response_format=parse
                        )
                        return response.choices[0].message.parsed
                    # 2) Two-pass approach: litellm -> openai parse
                    else:
                        if self.model in gemini_models:
                            client = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options={'api_version': 'v1alpha'})
                            response = await client.aio.models.generate_content(
                                model=self.model,
                                contents=messages[0]["content"]
                            )
                            unparsed_response =  response.text
                        elif self.model in anthropic_models:
                            client = anthropic.AsyncAnthropic(
                                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                            )
                            # Check if the message contains images
                            if isinstance(messages[0]["content"], list):
                                response = await client.messages.create(
                                    model=self.model,
                                    max_tokens=self.max_tokens,
                                    messages=[{
                                        "role": messages[0]["role"],
                                        "content": messages[0]["content"]
                                    }]
                                )
                            else:
                                response = await client.messages.create(
                                    model=self.model,
                                    max_tokens=self.max_tokens,
                                    messages=messages
                                )
                            unparsed_response = response.content[0].text.strip()
                        elif self.model in openai_models:
                            client = AsyncOpenAI()
                            if get_response:
                                response = await client.responses.create(
                                    model=self.model,
                                    input=messages
                                )
                                return response.output_text.strip()
                            else:
                                response = await client.chat.completions.create(
                                    model=self.model,
                                    messages=messages
                                )
                            unparsed_response = response.choices[0].message.content.strip()
                        elif self.model in deepseek_models:
                            client = AsyncOpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url="https://api.deepseek.com")
                            response = await client.chat.completions.create(
                                model=self.model,
                                messages=messages
                            )
                            unparsed_response = response.choices[0].message.content.strip()
                        else:
                            try:
                                # First pass (litellm)
                                litellm_response = await litellm.acompletion(
                                    model=self.model,
                                    messages=messages
                                )
                                unparsed_response = litellm_response["choices"][0]["message"]["content"].strip()
                            except Exception as e:
                                raise ValueError(f"Error calling litellm: {e}")
                        if parse:
                            # Second pass with openai parse
                            client = AsyncOpenAI()
                            response = await client.beta.chat.completions.parse(
                                model=parsing_model,
                                messages=[{
                                    "role": "user",
                                    "content": f"Given the following data, format it with the given response format: {unparsed_response}"
                                }],
                                response_format=parse
                            )
                            return response.choices[0].message.parsed
                        else:
                            return unparsed_response
                except Exception as e:
                    attempt += 1
                    if attempt < max_retries:
                        logging.warning(
                            f"Attempt {attempt} failed. Waiting 10 seconds before retry...\nError: {str(e)}"
                        )
                        await asyncio.sleep(10)
                    else:
                        logging.error(f"Failed after {max_retries} attempts. Error: {str(e)}")
                        return None

    async def process_single(self, prompt: Union[str, List[Dict[str, Any]]], parse: BaseModel = None, get_response=False) -> str:
        """
        Process a single prompt asynchronously.
        Prompt can be a string or a list of content blocks with images.
        """
        messages = self._format_message_with_images(prompt)
        return await self._call_api_async(messages, parse, get_response)

    def process_single_sync(self, prompt: Union[str, List[Dict[str, Any]]], parse: BaseModel = None, get_response=False) -> str:
        """
        Process a single prompt synchronously.
        Prompt can be a string or a list of content blocks with images.
        """
        messages = self._format_message_with_images(prompt)
        return self._call_api_sync(messages, parse, get_response)
        
    async def process_image(self, image_path_or_url: str, prompt: str, parse: BaseModel = None, get_response=True) -> str:
        """
        Process an image with a text prompt asynchronously.
        """
        try:
            # Encode the image
            base64_data, media_type = self._encode_image_to_base64(image_path_or_url)
            
            # Create content blocks based on model type
            if self.model in openai_models:
                 content = [
                    
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{base64_data}"
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            else: # Assume Anthropic/other format otherwise
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            
            # Process with the model
            return await self.process_single(content, parse, get_response)
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise
            
    def process_image_sync(self, image_path_or_url: str, prompt: str, parse: BaseModel = None, get_response=True) -> str:
        """
        Process an image with a text prompt synchronously.
        """
        try:
            # Encode the image
            base64_data, media_type = self._encode_image_to_base64(image_path_or_url)
            
            # Create content blocks based on model type
            if self.model in openai_models:
                 content = [ 
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{base64_data}"
                    },
                    {
                        "type": "input_text",
                        "text": prompt
                    }
                ]
            else: # Assume Anthropic/other format otherwise
                content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            
            # Process with the model
            return self.process_single_sync(content, parse, get_response)
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            raise

    async def process_batch(self, prompts: List[Union[str, List[Dict[str, Any]]]], parse: BaseModel = None) -> List[str]:
        """Process a batch of prompts concurrently."""
        tasks = []
        # Build the message array for each prompt
        for prompt in prompts:
            messages = self._format_message_with_images(prompt)
            tasks.append(messages)

        try:
            if self.model in openai_models:
                results = await self.batch_completions(tasks, parse=parse, get_response=True)
            else:
                results = await self.batch_completions(tasks, parse=parse)
                
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in result: {str(result)}")
                    processed_results.append(None)
                else:
                    processed_results.append(result)
            return processed_results
        except Exception as e:
            logger.error(f"Error in gather: {str(e)}")
            return [None] * len(prompts)

    async def batch_completions(self, prompts, parse=None, get_response=False):
        """
        Creates async tasks to call `_call_api_async` for each set of messages 
        (or use your direct litellm usage).
        """
        tasks = [
            asyncio.create_task(self._call_api_async(p, parse=parse, get_response=get_response)) 
            for p in prompts
        ]
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

