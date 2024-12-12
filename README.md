### Installation and Setup

To set up your virtual environment, you can use whatever suits you. In windows, you can use the `venv` package:
```
python -m venv .venv
.venv/Scripts/activate
```

Then, inside the **activated environment**, just install everything in `requirements.txt` using the following command:
```
pip install -r requirements.txt
```

Note, requirements are only tested with python 3.10.2, any deviation may require a different set of dependencies.