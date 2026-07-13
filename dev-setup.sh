#!/bin/sh -ex

# Set up a local development environment where we don't let
# vercel-worker's override our pinned vercel version.
# (Upstream fixes are in progress)
VERSION=$(cat .python-version)
cd backend
rm -rf .vercel/python/ .venv
uv sync
mkdir -p .vercel/python
cd .vercel/python/
cp -r ../../.venv/lib/python$VERSION/site-packages/vercel*.dist-info .
sed -i.bak '/\.\./d' vercel*.dist-info/RECORD
rm vercel*.dist-info/RECORD.bak
