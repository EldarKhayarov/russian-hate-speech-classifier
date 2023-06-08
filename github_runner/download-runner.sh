#!/bin/bash

mkdir actions-runner
cd actions-runner || exit

curl -o actions-runner-linux-arm64-2.304.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.304.0/actions-runner-linux-arm64-2.304.0.tar.gz
echo "34c49bd0e294abce6e4a073627ed60dc2f31eee970c13d389b704697724b31c6  actions-runner-linux-arm64-2.304.0.tar.gz" | shasum -a 256 -c
tar xzf ./actions-runner-linux-arm64-2.304.0.tar.gz
