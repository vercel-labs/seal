# seal

## answer style

be brief, use simple language.

## code guidelines

1. use [AI Elements](https://elements.ai-sdk.dev/docs.md) for all ui features. this is a collection of pre-built components spanning most of the ai application basics.
2. in python, import by module (unless it's `typing`) to improve namespacing and make it read to navigate code.
3. minimize the number of helper functions, prioritize locality of behavior.
4. keep apis as small as possible. keep public apis even smaller, try to shrink them to one function / object.
5. test file structure should mirror app's file structure, e.g. `agent/proto.py` -> `tests/agent/test_proto.py`. this helps project navigation a lot.

## project setup

1. use uv to manage python
2. use pnpm to manage typescript

## references

ai-python: .reference/ai-python
workflow for python: .reference/vercel-py
workflow for typescript: .reference/workflow
ai-elements: .reference/ai-elements
