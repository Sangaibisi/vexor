#!/bin/bash

set -e

echo "Installing Vexor dependencies..."

pip install -e ".[dev,crewai]"

echo "Done!"
