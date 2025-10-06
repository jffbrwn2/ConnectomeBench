"""Utility CLI for handling CAVEclient authentication tokens.

Usage examples:

1. Print instructions and URL to generate a new token:
   uv run python scripts/caveclient_token.py --request

2. Save a freshly generated token string:
   uv run python scripts/caveclient_token.py --save "<token>"

3. Show where the token is stored on disk (and whether it exists):
   uv run python scripts/caveclient_token.py --show-path
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from caveclient import CAVEclient

DEFAULT_SERVER = "https://global.daf-apis.com"
DEFAULT_DATASTACK = "minnie65_public"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Helper for requesting and storing authentication tokens required "
            "by CAVEclient/CloudVolume services."
        )
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=(
            "CAVE service base URL. Defaults to %(default)s. "
            "This should be the same server where you request tokens."
        ),
    )
    parser.add_argument(
        "--datastack",
        default=DEFAULT_DATASTACK,
        help=(
            "Datastack name used when printing existing token information. "
            "Does not affect token generation. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "--request",
        action="store_true",
        help=(
            "Print instructions (including the login URL) for generating a "
            "new token in the browser."
        ),
    )
    parser.add_argument(
        "--save",
        metavar="TOKEN",
        help="Persist a freshly generated token string to the standard location.",
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Print the expected token path and whether a token file exists.",
    )
    return parser


def construct_client(server: str) -> CAVEclient:
    """Instantiate a CAVEclient pointed at the provided server."""

    return CAVEclient(server_address=server)


def display_token_path(client: CAVEclient, datastack: str) -> None:
    token_path: Optional[Path] = None
    if hasattr(client.auth, "token_path"):
        token_path = Path(client.auth.token_path)

    if token_path is None:
        print(
            "Token path is unknown for this caveclient version. "
            "Typically tokens are stored under "
            "~/.cloudvolume/secrets/<host>-cave-secret.json."
        )
        return

    exists = token_path.exists()
    status = "exists" if exists else "does not exist"
    print(f"Token path: {token_path} ({status})")

    if exists:
        try:
            masked = Path(token_path).read_text().strip()
            if masked:
                masked = masked[:6] + "..." + masked[-4:]
                print(f"Existing token (masked): {masked}")
        except OSError as exc:
            print(f"Could not read token file: {exc}")


def request_token_instructions(client: CAVEclient) -> None:
    """Display the standard instructions for generating a new token."""

    # The caveclient helper prints instructions to stdout on its own.
    client.auth.get_new_token()


def save_token(client: CAVEclient, token: str) -> None:
    """Persist the provided token using caveclient's helper."""

    client.auth.save_token(token=token)
    token_path = getattr(client.auth, "token_path", None)
    if token_path:
        print(f"Saved token to {token_path}.")
    else:
        print("Saved token (token path not reported by this caveclient version).")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    client = construct_client(args.server)

    any_action = False

    if args.show_path:
        display_token_path(client, args.datastack)
        any_action = True

    if args.request:
        request_token_instructions(client)
        any_action = True

    if args.save:
        save_token(client, args.save)
        any_action = True

    if not any_action:
        parser.print_help()


if __name__ == "__main__":
    main()
