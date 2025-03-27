import { Anthropic } from "@anthropic-ai/sdk"
import OpenAI, { AzureOpenAI } from "openai"
import { withRetry } from "../retry"
import { ApiHandlerOptions, azureOpenAiDefaultApiVersion, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import { ApiHandler } from "../index"
import { convertToOpenAiMessages } from "../transform/openai-format"
import { ApiStream } from "../transform/stream"
import { convertToR1Format } from "../transform/r1-format"
import { ChatCompletionReasoningEffort } from "openai/resources/chat/completions.mjs"

export class OpenAiHandler implements ApiHandler {
	private options: ApiHandlerOptions
	private client: OpenAI
	private availableModels: string[] = []

	constructor(options: ApiHandlerOptions) {
		this.options = options
		const apiKey = this.options.openAiApiKey || ''; // Make API key explicitly optional
		// Azure API shape slightly differs from the core API shape: https://github.com/openai/openai-node?tab=readme-ov-file#microsoft-azure-openai
		// Use azureApiVersion to determine if this is an Azure endpoint, since the URL may not always contain 'azure.com'
		if (this.options.azureApiVersion || this.options.openAiBaseUrl?.toLowerCase().includes("azure.com")) {
			this.client = new AzureOpenAI({
				baseURL: this.options.openAiBaseUrl,
				apiKey: apiKey, // API key is optional
				apiVersion: this.options.azureApiVersion || azureOpenAiDefaultApiVersion,
			})
		} else {
			this.client = new OpenAI({
				baseURL: this.options.openAiBaseUrl,
				apiKey: apiKey, // API key is optional
			})
		}

		// Initialize the list of available models
		this.initializeModelList();
	}

	private async initializeModelList() {
		try {
			this.availableModels = await this.listModels();
			console.log(`Retrieved ${this.availableModels.length} models from the server`);
		} catch (error: any) {
			console.error("Failed to retrieve models during initialization:", error.message);
			this.availableModels = [];
		}
	}

	@withRetry()
	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		// If model ID is not specified, try to get the first available model from the cached list
		let modelId = this.options.openAiModelId ?? ""
		
		if (!modelId) {
			// Use cached models if available, otherwise refresh the list
			if (this.availableModels.length === 0) {
				try {
					this.availableModels = await this.listModels();
				} catch (error: any) {
					console.error("Failed to retrieve models:", error.message);
				}
			}
			
			if (this.availableModels.length > 0) {
				modelId = this.availableModels[0];
				console.log(`Using first available model: ${modelId}`);
			} else {
				throw new Error("No models available from the OpenAI-compatible server");
			}
		}
		
		const isDeepseekReasoner = modelId.includes("deepseek-reasoner")
		const isR1FormatRequired = this.options.openAiModelInfo?.isR1FormatRequired ?? false
		const isO3Mini = modelId.includes("o3-mini")

		let openAiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "system", content: systemPrompt },
			...convertToOpenAiMessages(messages),
		]
		let temperature: number | undefined = this.options.openAiModelInfo?.temperature ?? openAiModelInfoSaneDefaults.temperature
		let reasoningEffort: ChatCompletionReasoningEffort | undefined = undefined
		let maxTokens: number | undefined

		if (this.options.openAiModelInfo?.maxTokens && this.options.openAiModelInfo.maxTokens > 0) {
			maxTokens = Number(this.options.openAiModelInfo.maxTokens)
		} else {
			maxTokens = undefined
		}

		if (isDeepseekReasoner || isR1FormatRequired) {
			openAiMessages = convertToR1Format([{ role: "user", content: systemPrompt }, ...messages])
		}

		if (isO3Mini) {
			openAiMessages = [{ role: "developer", content: systemPrompt }, ...convertToOpenAiMessages(messages)]
			temperature = undefined // does not support temperature
			reasoningEffort = (this.options.o3MiniReasoningEffort as ChatCompletionReasoningEffort) || "medium"
		}

		const stream = await this.client.chat.completions.create({
			model: modelId,
			messages: openAiMessages,
			temperature,
			max_tokens: maxTokens,
			reasoning_effort: reasoningEffort,
			stream: true,
			stream_options: { include_usage: true },
		})
		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta
			if (delta?.content) {
				yield {
					type: "text",
					text: delta.content,
				}
			}

			if (delta && "reasoning_content" in delta && delta.reasoning_content) {
				yield {
					type: "reasoning",
					reasoning: (delta.reasoning_content as string | undefined) || "",
				}
			}

			if (chunk.usage) {
				yield {
					type: "usage",
					inputTokens: chunk.usage.prompt_tokens || 0,
					outputTokens: chunk.usage.completion_tokens || 0,
				}
			}
		}
	}

	async listModels(): Promise<string[]> {
		try {
			const response = await this.client.models.list();
			return response.data.map((model: any) => model.id);
		} catch (error: any) { // Explicitly type error as any or Error
			console.error("Error fetching model list:", error.message);
			return [];
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.openAiModelId ?? "",
			info: this.options.openAiModelInfo ?? openAiModelInfoSaneDefaults,
		}
	}

	/**
	 * Gets the list of models available from the OpenAI-compatible server
	 * This can be used by the UI to present a model selection dropdown to the user
	 * @returns The list of model IDs available from the server
	 */
	getAvailableModels(): string[] {
		return this.availableModels;
	}
}
