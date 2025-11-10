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
#host <- "http://ollama.runai-shared.svc.cluster.local"

# on HPC (for this training session)
host <- "http://ollama-training.runai-shared.svc.cluster.local"


test_connection(url=host)  # test connection to Ollama server
# if you see "Ollama local server not running or wrong server," Ollama app/server isn't running

# download a model
pull("gemma3:4b", host=host, stream = TRUE)  # download a model.

# generate a response/text based on a prompt; returns an httr2 response by default
# the generate function you can think of as autocomplete, given some text it will
# continue to predict token by token and return the end result.
resp <- generate("gemma3:4b", "tell me a 5-word story", host=host)
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
txt <- generate("gemma3:4b", "tell me a 5-word story", output = "text", host= host)

# list available models (models already pulled/downloaded)
list_models(host=host)
#name    size parameter_size quantization_level            modified
#1  gemma3:12b  8.1 GB          12.2B             Q4_K_M 2025-11-10T22:20:54
#2   gemma3:4b  3.3 GB           4.3B             Q4_K_M 2025-11-10T22:21:30

#
# Stuctured Outputs
#

# Stuctured outputs forces a LLM to products a structured output which can be
# parsed. This is particularly useful in extracting information from free-text
# inputs.

# First we define the structure we want the LLM to respond with.
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

# Then call it specifying the model, prompt and structure
generate("gemma3:4b", "tell me about London", output = "structured", format = format, host=host)


# Let's give it a go, extracting information from an export of ZSL's Wikipedia page.
# (https://en.wikipedia.org/wiki/Zoological_Society_of_London)
#
# Bonus points, try extracting the contents of a paper and see if you can get the LLM
# to identify which species are mentioned in it
#
text <- readChar("data/zsl_wiki.md", file.info("data/zsl_wiki.md")$size)
text

# This time we'll use a chat concept, instead of 'generate', we'll use a chat
# endpoint which forces the model into a chat like template.
#
# There are three types of messages we can generally use:
#
# system: messages that guide how the model should respond. e.g. pretend to be a character or specify tone
# user: chat messages a user has written
# assistant: previous chat messages the LLM has written
#
# So let's create a message thread...
messages <- create_messages(
  create_message("You are a friendly ZSL historian, use the information provided to respond to users questions", role="system"),
  create_message(paste("Content: ", text), role="system"),
  create_message("What year was ZSL formed?")
)


# We then call the model using chat() and print out the response with cat().
cat(
  chat(
    "gemma3:4b", messages, output = "text", host = host
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
  "gemma3:4b", messages, output = "structured", host = host, format = format
)
response


# You might find that the model hallucinates (making things up or generally
# responding with the wrong information), particularly problematic with smaller
# models, try with larger model:

pull("gemma3:12b", host=host, stream = TRUE)  # download a model.

response <-  chat(
  "gemma3:12b", messages, output = "structured", host = host, format = format
)
response

