#!/bin/bash
while read k; do
    pip install "$k"
done < requirements.txt