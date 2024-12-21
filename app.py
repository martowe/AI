import datetime
import bcrypt
import openai
import requests
import anthropic
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from extensions import db, login_manager
from models import User, SingleModelMessage  # <-- updated import
from config import Config


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    login_manager.init_app(app)
    with app.app_context():
        db.create_all()
    return app


app = create_app()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("User already exists.", "error")
            return redirect(url_for('register'))

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(email=email, password=hashed_password.decode('utf-8'))
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful. Please login.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials.", "error")
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        # Update API keys
        current_user.chatgpt_api_key = request.form.get('chatgpt_api_key')
        current_user.gemini_api_key = request.form.get('gemini_api_key')
        current_user.xai_api_key = request.form.get('xai_api_key')
        current_user.claude_api_key = request.form.get('claude_api_key')

        # Update enable/disable flags
        current_user.enable_chatgpt = (request.form.get('enable_chatgpt') == 'on')
        current_user.enable_gemini = (request.form.get('enable_gemini') == 'on')
        current_user.enable_xai = (request.form.get('enable_xai') == 'on')
        current_user.enable_claude = (request.form.get('enable_claude') == 'on')

        db.session.commit()
        flash("Profile updated successfully!", "success")
    return render_template('profile.html')


@app.route('/compare', methods=['GET', 'POST'])
@login_required
def compare():
    response_chatgpt = None
    response_gemini = None
    response_xai = None
    response_claude = None

    if request.method == 'POST':
        prompt = request.form.get('prompt')
        chatgpt_model = request.form.get('chatgpt_model', 'gpt-3.5-turbo')
        gemini_model = request.form.get('gemini_model', 'gemini-1.5-pro')
        xai_model = request.form.get('xai_model', 'grok-2')  # Corrected default model
        claude_model = request.form.get('claude_model', 'claude-3-sonnet-20240229')

        # Enforce usage limit for all users
        today = datetime.date.today()
        if current_user.last_usage_date != today:
            current_user.daily_usage_count = 0
            current_user.last_usage_date = today
            db.session.commit()

        if current_user.daily_usage_count >= app.config['FREE_TIER_DAILY_LIMIT']:
            flash("You have reached your daily limit.", "error")
            return render_template('compare.html')

        current_user.daily_usage_count += 1
        db.session.commit()

        def get_api_key(user_key, config_key):
            return user_key if user_key else app.config.get(config_key)

        # 1) ChatGPT
        if current_user.enable_chatgpt:
            chatgpt_api_key = get_api_key(current_user.chatgpt_api_key, 'CHATGPT_FREE_API_KEY')
            if chatgpt_api_key:
                try:
                    openai.api_key = chatgpt_api_key
                    chat_response = openai.ChatCompletion.create(
                        model=chatgpt_model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response_chatgpt = chat_response.choices[0].message["content"]
                except Exception as e:
                    response_chatgpt = f"Error using ChatGPT API: {str(e)}"

        # 2) Gemini
        if current_user.enable_gemini:
            gemini_api_key = get_api_key(current_user.gemini_api_key, 'GEMINI_FREE_API_KEY')
            if gemini_api_key:
                try:
                    gemini_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_api_key}"
                    payload = {
                        "contents": [
                            {
                                "parts": [{"text": prompt}]
                            }
                        ]
                    }
                    headers = {"Content-Type": "application/json"}
                    r = requests.post(gemini_endpoint, json=payload, headers=headers)
                    r.raise_for_status()

                    resp_json = r.json()
                    candidates = resp_json.get("candidates", [])
                    if candidates and len(candidates) > 0:
                        content = candidates[0].get("content", {})
                        if content:
                            parts = content.get("parts", [])
                            if parts and len(parts) > 0:
                                response_gemini = parts[0].get("text", "No text in response")
                            else:
                                response_gemini = "No text parts in Gemini response"
                        else:
                            response_gemini = "No content in Gemini response"
                    else:
                        response_gemini = "No candidates in Gemini response"
                except requests.exceptions.RequestException as e:
                    response_gemini = f"Error calling Gemini API: {str(e)}"

        # 3) xAI
        if current_user.enable_xai:
            xai_api_key = get_api_key(current_user.xai_api_key, 'XAI_FREE_API_KEY')
            if xai_api_key:
                try:
                    xai_endpoint = "https://api.x.ai/v1/chat/completions"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {xai_api_key}"
                    }
                    payload = {
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "model": xai_model,
                        "stream": False,
                        "temperature": 0
                    }
                    resp = requests.post(xai_endpoint, headers=headers, json=payload)
                    resp.raise_for_status()

                    resp_json = resp.json()
                    choices = resp_json.get("choices", [])
                    if choices:
                        response_xai = choices[0].get("message", {}).get("content", "No xAI content found.")
                    else:
                        response_xai = "No xAI choices returned."
                except requests.exceptions.RequestException as e:
                    response_xai = f"Error calling xAI API: {str(e)}"

        # 4) Claude
        if current_user.enable_claude:
            claude_api_key = get_api_key(current_user.claude_api_key, 'CLAUDE_FREE_API_KEY')
            if claude_api_key:
                try:
                    client = anthropic.Anthropic(api_key=claude_api_key)
                    message = client.messages.create(
                        model=claude_model,
                        max_tokens=1024,
                        temperature=0,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    response_claude = message.content[0].text
                except anthropic.APIStatusError as e:
                    response_claude = f"Error calling Claude API: {e.status_code} - {e.message}"
                except anthropic.APIConnectionError as e:
                    response_claude = f"Error connecting to Claude API: {e}"
                except anthropic.AuthenticationError as e:
                    response_claude = f"Claude API Authentication Error: {e}"
                except anthropic.RateLimitError as e:
                    response_claude = f"Claude API Rate Limit Error: {e}"
                except Exception as e:
                    response_claude = f"Unexpected error calling Claude API: {str(e)}"

    return render_template(
        'compare.html',
        response_chatgpt=response_chatgpt,
        response_gemini=response_gemini,
        response_xai=response_xai,
        response_claude=response_claude
    )


# --- NEW CODE START ---
@app.route('/single_model_chat', methods=['GET', 'POST'])
@login_required
def single_model_chat():
    """
    A route that allows chatting with exactly one model at a time.
    Conversation history is stored and displayed.
    """
    # Enforce usage limit for all users
    today = datetime.date.today()
    if current_user.last_usage_date != today:
        current_user.daily_usage_count = 0
        current_user.last_usage_date = today
        db.session.commit()

    if current_user.daily_usage_count >= app.config['FREE_TIER_DAILY_LIMIT']:
        flash("You have reached your daily limit.", "error")
        return render_template('single_model_chat.html', conversation=[])

    conversation = []
    if request.method == 'POST':
        selected_model = request.form.get('model_choice')
        prompt = request.form.get('prompt', '').strip()

        if prompt:
            # Increment usage
            current_user.daily_usage_count += 1
            db.session.commit()

            # Helper to get the correct API key for the selected model
            def get_api_key_for_model(model_name):
                if model_name == 'ChatGPT':
                    return current_user.chatgpt_api_key if current_user.chatgpt_api_key else app.config.get(
                        'CHATGPT_FREE_API_KEY')
                elif model_name == 'Gemini':
                    return current_user.gemini_api_key if current_user.gemini_api_key else app.config.get(
                        'GEMINI_FREE_API_KEY')
                elif model_name == 'xAI':
                    return current_user.xai_api_key if current_user.xai_api_key else app.config.get('XAI_FREE_API_KEY')
                elif model_name == 'Claude':
                    return current_user.claude_api_key if current_user.claude_api_key else app.config.get(
                        'CLAUDE_FREE_API_KEY')
                return None

            # Create a new record for the user's message
            user_msg = SingleModelMessage(
                user_id=current_user.id,
                model_name=selected_model,
                role='user',
                content=prompt
            )
            db.session.add(user_msg)
            db.session.commit()

            response_text = "No response or disabled model."
            # If the model is enabled, call its API
            if selected_model == 'ChatGPT' and current_user.enable_chatgpt:
                api_key = get_api_key_for_model('ChatGPT')
                if api_key:
                    try:
                        openai.api_key = api_key
                        chat_response = openai.ChatCompletion.create(
                            model='gpt-3.5-turbo',
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_text = chat_response.choices[0].message["content"]
                    except Exception as e:
                        response_text = f"Error using ChatGPT API: {str(e)}"

            elif selected_model == 'Gemini' and current_user.enable_gemini:
                api_key = get_api_key_for_model('Gemini')
                if api_key:
                    try:
                        gemini_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=" + api_key
                        payload = {
                            "contents": [
                                {
                                    "parts": [{"text": prompt}]
                                }
                            ]
                        }
                        headers = {"Content-Type": "application/json"}
                        r = requests.post(gemini_endpoint, json=payload, headers=headers)
                        r.raise_for_status()

                        resp_json = r.json()
                        candidates = resp_json.get("candidates", [])
                        if candidates and len(candidates) > 0:
                            content = candidates[0].get("content", {})
                            if content:
                                parts = content.get("parts", [])
                                if parts and len(parts) > 0:
                                    response_text = parts[0].get("text", "No text in response")
                                else:
                                    response_text = "No text parts in Gemini response"
                            else:
                                response_text = "No content in Gemini response"
                        else:
                            response_text = "No candidates in Gemini response"
                    except requests.exceptions.RequestException as e:
                        response_text = f"Error calling Gemini API: {str(e)}"

            elif selected_model == 'xAI' and current_user.enable_xai:
                api_key = get_api_key_for_model('xAI')
                if api_key:
                    try:
                        xai_endpoint = "https://api.x.ai/v1/chat/completions"
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}"
                        }
                        payload = {
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "model": "grok-2",
                            "stream": False,
                            "temperature": 0
                        }
                        resp = requests.post(xai_endpoint, headers=headers, json=payload)
                        resp.raise_for_status()

                        resp_json = resp.json()
                        choices = resp_json.get("choices", [])
                        if choices:
                            response_text = choices[0].get("message", {}).get("content", "No xAI content found.")
                        else:
                            response_text = "No xAI choices returned."
                    except requests.exceptions.RequestException as e:
                        response_text = f"Error calling xAI API: {str(e)}"

            elif selected_model == 'Claude' and current_user.enable_claude:
                api_key = get_api_key_for_model('Claude')
                if api_key:
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        message = client.messages.create(
                            model='claude-2',
                            max_tokens=1024,
                            temperature=0,
                            messages=[
                                {"role": "user", "content": prompt}
                            ]
                        )
                        response_text = message.content[0].text
                    except anthropic.APIStatusError as e:
                        response_text = f"Error calling Claude API: {e.status_code} - {e.message}"
                    except anthropic.APIConnectionError as e:
                        response_text = f"Error connecting to Claude API: {e}"
                    except anthropic.AuthenticationError as e:
                        response_text = f"Claude API Authentication Error: {e}"
                    except anthropic.RateLimitError as e:
                        response_text = f"Claude API Rate Limit Error: {e}"
                    except Exception as e:
                        response_text = f"Unexpected error calling Claude API: {str(e)}"

            # Save assistant reply
            assistant_msg = SingleModelMessage(
                user_id=current_user.id,
                model_name=selected_model,
                role='assistant',
                content=response_text
            )
            db.session.add(assistant_msg)
            db.session.commit()

    chosen_model = request.form.get('model_choice') or 'ChatGPT'  # default
    if chosen_model not in ['ChatGPT', 'Gemini', 'xAI', 'Claude']:
        chosen_model = 'ChatGPT'

    conversation = SingleModelMessage.query.filter_by(user_id=current_user.id, model_name=chosen_model).order_by(
        SingleModelMessage.timestamp.asc()).all()
    return render_template('single_model_chat.html', conversation=conversation, chosen_model=chosen_model)


# --- NEW CODE END ---

# --- NEW CODE START ---
@app.route('/donate')
def donate():
    """
    A route displaying donation options: Revolut, Bank IBAN, and PayPal.
    """
    return render_template('donate.html')


# --- NEW CODE END ---

if __name__ == '__main__':
    app.run(debug=True)
