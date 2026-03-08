# mamba-vs-gpt

  1. Setup
``` bash
  curl -fsSL https://raw.githubusercontent.com/tcoyne1729/mamba-vs-gpt/refs/heads/main/runners/setup.sh | bash
                                                                                        ```
  2. Enter the repo dir and activate the venv

  ``` bash
  cd mamba-vs-gpt
  source .venv/bin/activate
  ```
  3. Launch the dev run
``` bash
  DEV_RUN=1 python gpt_train.py
```

  Or launch the prod run with

``` bash
python gpt_train.py
```
