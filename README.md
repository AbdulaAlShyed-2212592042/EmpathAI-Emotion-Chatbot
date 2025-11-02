# EmpathAI-Emotion-Chatbot
An emotion-aware AI chatbot that combines a fine-tuned GoEmotions model with ChatGPT API to generate empathetic, context-sensitive responses.

## Conversational backbone

For the conversational backbone of the system we leverage OpenAI's ChatGPT free tier as our large-language-model component during prototyping and early deployment. According to OpenAI's free-tier documentation, users may access models such as GPT-4o (and related variants) with usage limits and reduced throughput compared to paid tiers. This provides an easy, low-cost integration path for development while we focus on emotion recognition, prompt engineering, and backend orchestration.

In practice, user text is first passed to our emotion-recognition module which outputs an emotion label (or labels). That label is appended to the user's text via prompt engineering — for example: "User appears sad; please respond with empathy and helpfulness." The composed prompt guides ChatGPT to produce responses that are emotionally aware, contextually relevant, and conversational.

Because free-tier access has usage caps and may not support advanced fine‑tuning or high-throughput production loads, the system is built to be modular: the emotion model and prompt builder are separate from the conversational LLM. This makes it straightforward to migrate to a self-hosted model (for example, Llama 2-Chat or another open model) or a paid OpenAI plan when custom fine-tuning, higher throughput, or removal of usage limits is required.

This hybrid, modular approach lets us prototype quickly for free while keeping a clear upgrade path to higher-capacity or self-hosted LLMs as needs evolve.

