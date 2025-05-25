from typing import List, Union, Optional, Dict

# Subtypes

class TextContent:
    def __init__(self, text: str):
        # Content type is text
        self.type = "text"
        self.text = text

class ImageContentPart:
    def __init__(self, url: str, detail: Optional[str] = "auto"):
        # Content type is image_url
        self.type = "image_url"
        self.image_url = {
            "url": url,  # URL or base64 encoded image data
            "detail": detail  # Optional, defaults to "auto"
        }

ContentPart = Union[TextContent, ImageContentPart]

class Message:
    def __init__(self, role: str, content: Union[str, List[ContentPart]], tool_call_id: Optional[str] = None, name: Optional[str] = None):
        # Role can be "user", "assistant", "system", or "tool"
        self.role = role
        # ContentParts are only for the "user" role
        self.content = content
        # Tool call ID for "tool" role
        self.tool_call_id = tool_call_id
        # If "name" is included, it will be prepended like this for non-OpenAI models: `{name}: {content}`
        self.name = name

class FunctionDescription:
    def __init__(self, name: str, parameters: dict, description: Optional[str] = None):
        # Description of the function
        self.description = description
        # Name of the function
        self.name = name
        # JSON Schema object for parameters
        self.parameters = parameters

class Tool:
    def __init__(self, function: FunctionDescription):
        # Tool type is function
        self.type = "function"
        self.function = function

ToolChoice = Union[str, Dict[str, Dict[str, str]]]

from typing import List, Optional, Union

class ProviderPreferences:
    def __init__(
        self,
        allow_fallbacks: Optional[bool] = True,
        require_parameters: Optional[bool] = False,
        data_collection: Optional[str] = "deny",
        order: Optional[List[str]] = ["OpenAI", "Anthropic", "Google"],
        ignore: Optional[List[str]] = None,
        quantizations: Optional[List[str]] = None
    ):
        # Whether to allow backup providers to serve requests
        # - True: (default) when the primary provider (or your custom providers in "order") is unavailable, use the next best provider.
        # - False: use only the primary/custom provider, and return the upstream error if it's unavailable.
        self.allow_fallbacks = allow_fallbacks

        # Whether to filter providers to only those that support the parameters you've provided.
        # If this setting is omitted or set to False, then providers will receive only the parameters they support, and ignore the rest.
        self.require_parameters = require_parameters

        # Data collection setting.
        # - "allow": (default) allow providers which store user data non-transiently and may train on it
        # - "deny": use only providers which do not collect user data.
        self.data_collection = data_collection

        # An ordered list of provider names. The router will attempt to use the first provider in the subset of this list that supports your requested model, and fall back to the next if it is unavailable.
        # If no providers are available, the request will fail with an error message.
        self.order = order

        # List of provider names to ignore. If provided, this list is merged with your account-wide ignored provider settings for this request.
        self.ignore = ignore

        # A list of quantization levels to filter the provider by.
        self.quantizations = quantizations

        # Example usage:
        # provider_prefs = ProviderPreferences(
        #     allow_fallbacks=True,
        #     require_parameters=False,
        #     data_collection="allow",
        #     order=["OpenAI", "Anthropic"],
        #     ignore=["Google"],
        #     quantizations=["int8", "fp16"]
        # )

class Request:
    def __init__(
        self,
        messages: Optional[List[Message]] = None,
        prompt: Optional[str] = None,
        model: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[ToolChoice] = None,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        top_logprobs: int = 0,
        min_p: Optional[float] = None,
        top_a: Optional[float] = None,
        prediction: Optional[Dict[str, str]] = None,
        transforms: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        route: Optional[str] = None,
        provider: Optional[ProviderPreferences] = None
    ):
        # Either "messages" or "prompt" is required
        self.messages = messages
        self.prompt = prompt
        # If "model" is unspecified, uses the user's default
        self.model = model
        # Allows to force the model to produce specific output format
        self.response_format = response_format
        self.stop = stop
        # Enable streaming
        self.stream = stream
        # Range: [1, context_length)
        self.max_tokens = max_tokens
        # Range: [0, 2]
        self.temperature = temperature
        # Tool calling
        self.tools = tools
        self.tool_choice = tool_choice
        # Advanced optional parameters
        # Integer only
        self.seed = seed
        # Range: (0, 1]
        self.top_p = top_p
        # Range: [1, Infinity) Not available for OpenAI models
        self.top_k = top_k
        # Range: [-2, 2]
        self.frequency_penalty = frequency_penalty
        # Range: [-2, 2]
        self.presence_penalty = presence_penalty
        # Range: (0, 2]
        self.repetition_penalty = repetition_penalty
        self.logit_bias = logit_bias
        # Integer only
        self.top_logprobs = top_logprobs
        # Range: [0, 1]
        self.min_p = min_p
        # Range: [0, 1]
        self.top_a = top_a
        # Reduce latency by providing the model with a predicted output
        self.prediction = prediction
        # OpenRouter-only parameters
        self.transforms = transforms
        self.models = models
        self.route = route
        self.provider = provider

# Assuming ProviderPreferences is defined elsewhere