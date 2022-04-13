ConceptNet Service for GreaseLM

## How to use

Build a docker image:
```docker build -t conceptnet-service -f Dockerfile .```

Run docker container: ```docker run -d -p 8080:8080 --gpus all conceptnet-service```

Test the service:
```curl -X POST http://localhost:8080/csqa/ -H "Content-Type: application/json" -d '{"answerKey": "B", "id": "61fe6e879ff18686d7552425a36344c8", "question": {"question_concept": "people", "choices": [{"label": "A", "text": "race track"}, {"label": "B", "text": "populated areas"}, {"label": "C", "text": "the desert"}, {"label": "D", "text": "apartment"}, {"label": "E", "text": "roadblock"}], "stem": "Sammy wanted to go to where the people were.  Where might he go?"}}'```

## License

This project is licensed under the terms of the MIT license.
