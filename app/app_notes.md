# loader folder 
- will contain the modeul for smart parser which will load the raw document convert it into document structure (page_content , metadata) accourding to the file 
-  for pdf we have pypdf
- html beautful spop
-- suggest differnt type of loader other than this and most popular and standart as well
-- imp from tabluear data - text to sql
-- suggest similar for excel ,pptx and markdown etc popular file extension (create a note )


-- what is parser? - converting one file type to another 

# chunking folder 
 - semantic chunking (semantic similar sentance are merge into one chunk )
 - layout level chunking 
 - context basesd chunking
 
# process all the complete ingestion pipelines

# services folder 
embeddding - > used to create embedding vector out of chunks

# commmand 
uvicorn app.main_:app --reload --port 8000

## streamlit
streamlit run ui/app.py

# Guardrails : (rails - rules and regularation)
## NeMo Guardrails -> by nvidia : open source used in development
 -  useing similarity for checking the intent of the query 
 - if the question is a threat 
 - "co" colang language by nvidia (neither english nor programming - little bit of both )
 - we define a rail 
    - define a user - give it a varaible name (eg ask offtopic)
    - define a bot, - give it a variable name (eg refuse offtopic)
    - define a flow - guard offtopic (if user is going offtopic then the bot can refuse offtopic)

    - user - (give example )
    - sentance embedding compare with define user example
    - for similarity it uses faseembeder (vector stor)
    - use ai to check if it offtopic

- todo : vector store vs vector db
## llama guardrain 
## AWS Bedrock -> cloud based sits on top of llm - industry / production 