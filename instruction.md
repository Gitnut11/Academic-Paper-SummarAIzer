# Things to do
- deal neo4j locally
- handle logging
- print
- find a way to minimize torch


> Put the `.env` file outside in the project folder
## Note
`NEO4J_URI="bolt://neo4j:7687"`\
`NEO4J_USERNAME="neo4j"`

## Run with docker
> Remember to define torch+ cpu or cuda version, to narrow down the installing time -- Using torch+cpu here ~ 330s
- For the first time:
```
docker-compose -p myapp up --build
```
- 2nd or more
    - `-d` tag can be added to let the docker run in the background
```
docker-compose up
```
- Run `docker-compose down` to stop it.



## Run locally
### Start Neo4j
```bash
docker run -d --name neo4j-server --env-file config/.env -p 7474:7474 -p 7687:7687 -e "NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}" neo4j:5.8
```

### Install dependencies
### Run backend
```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run frontend
```
streamlit run main.py
```

## Ports
- Neo4j Browser → http://localhost:7474
- FastAPI UI → http://localhost:8000/docs
- Streamlit → http://localhost:8501