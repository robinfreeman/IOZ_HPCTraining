#
# LLMs
#


# Install and load the ollamar R library. This is an API client to the [Ollama](https://ollama.com/)
# model runner and has capability to run a variety of models (https://ollama.com/search).
install.packages("ollamar")
library(ollamar)

# Usually Ollama would run on your local machine and would be the default host.
# However we have a shared instance running on the HPC allowing you to utilise larger
# models on BIG GPUs! So you can run larger more cablable models faster.
#
# Any workload can access the Ollama instance at http://ollama.runai-shared.svc.cluster.local.

# default / local:
#host <- "http://127.0.0.1:11434"

# on HPC:
host <- "http://ollama.runai-shared.svc.cluster.local"

test_connection(url=host)  # test connection to Ollama server
# if you see "Ollama local server not running or wrong server," Ollama app/server isn't running

# download a model
pull("gemma3", host=host)  # download a model.

# See what models exists
list_models(host = host)

# generate a response/text based on a prompt; returns an httr2 response by default
resp <- generate("gemma3", "tell me a 5-word story", host=host)
resp

#' interpret httr2 response object
#' <httr2_response>
#' POST http://127.0.0.1:11434/api/generate  # endpoint
#' Status: 200 OK  # if successful, status code should be 200 OK
#' Content-Type: application/json
#' Body: In memory (414 bytes)

# get just the text from the response object
resp_process(resp, "text")
# get the text as a tibble dataframe
resp_process(resp, "df")

# alternatively, specify the output type when calling the function initially
txt <- generate("gemma3", "tell me a 5-word story", output = "text", host= host)

# list available models (models already pulled/downloaded)
list_models(host=host)
#name    size parameter_size quantization_level            modified
#1               codegemma:7b    5 GB             9B               Q4_0 2024-07-27T23:44:10
#2            llama3.1:latest  4.7 GB           8.0B               Q4_0 2024-07-31T07:44:33

#
# Stuctured Outputs
#

# Stuctured outputs forces a LLM to products a structured output which can be
# parsed. This is particularly useful in extracting information from free-text
# inputs.

format <- list(
  type = "object",
  properties = list(
    city = list(type = "string"),
    country = list(type= "string"),
    language = list(type="string"),
    currency = list(type="string")
  ),
  required = list("city", "country", "language", "currency")
)

generate("gemma3", "tell me about London", output = "structured", format = format, host=host)


# Let's give it a go, extracting information from an export of ZSL's Wikipedia page.
# (https://en.wikipedia.org/wiki/Zoological_Society_of_London)
#
# Bonus points, try extracting the contents of a paper and see if you can get the LLM
# to identify which species are mentioned in it
#
text <- readChar("data/zsl_wiki.md", file.info("data/zsl_wiki.md")$size)
text


messages <- create_messages(
  create_message("You are a friendly ZSL historian, use the information provided to respond to users questions", role="system"),
  create_message(paste("Content: ", text), role="system"),
  create_message("What year was ZSL formed?")
)

cat(
  chat(
    "gemma3", messages, output = "text", host = host
  )
)

# That's given us a free text answer, let's get a structured output!


format <- list(
  type = "object",
  properties = list(
    year = list(type = "number")
  ),
  required = list("year")
)


response <-  chat(
    "gemma3", messages, output = "structured", host = host, format = format
)
response
