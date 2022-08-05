## Automate sail report with python

This is a project for show how we can automate a sail report using python and sql langueages.

## Setup

Clone the repository

```
git clone https://github.com/axelluguercio/data-sails-reporting.git
```

Install the requirements

```
pip3 install -r requirements.txt
```

### Create the .env file

The .env will store all secrets such as the crendentials and ip to connect to the db server, as the start and end date which you want to extract the report's information.

Once you filled the .env with all required secrets you can setup the env.

Example

```
DB="YOUR ip-FQDM and database location"
```

```
python3 -m venv env
source env/bin/activate
```

## Run

Run the automated script

```
chmod +x script.sh
./script.sh
```
