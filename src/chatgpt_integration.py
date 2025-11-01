import openai

# Set your OpenAI API key
openai.api_key = 'your_api_key_here'


def generate_emotion_aware_prompt(user_input, emotion):
    """
    Generate a prompt for the ChatGPT API based on the user's input and identified emotion.
    """
    return f"User is feeling {emotion}. Respond empathetically to: {user_input}"


def get_empathetic_response(user_input, emotion):
    """
    Get an empathetic response from the ChatGPT API.
    """
    prompt = generate_emotion_aware_prompt(user_input, emotion)
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message['content']


# Example usage
if __name__ == '__main__':
    user_input = "I'm feeling really sad today."
    emotion = "sad"
    response = get_empathetic_response(user_input, emotion)
    print(response)