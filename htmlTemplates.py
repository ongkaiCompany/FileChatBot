
css = '''
<style>
body {
    font-family: 'Open Sans', sans-serif;
}

.chat-message {
    margin: 20px;
    display: flex;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s, transform 0.3s;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    min-height: 60px;
}

.chat-message.user {
    color: black;
    border-radius: 20px;
    flex-direction: row-reverse;
}

.chat-message.bot {
    border-radius: 10px;
}

.chat_input{
    border: 1px solid black;
    }

.chat-message .avatar {
    width: 10%; /* Adjust to a flexible unit, like a percentage */
    height: 60px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 10px;
    transition: transform 0.3s;
}

.chat-message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.chat-message:hover {
    background-color: #f0f0f0;
}

.avatar:hover {
    transform: scale(1.1);
}

.chat-message .message {
    border: 1px solid transparent;
    padding: 10px;
    border-radius: 10px;
    background: linear-gradient(to bottom, #ffffff, #f0f0f0);
    width: 90%; /* Adjust to complement the avatar width */
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.chat-message {
    animation: fadeIn 0.5s ease-out;
}
</style>
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar" style="aspect-ratio: 1 / 1;">
        <img src="https://i.ibb.co/DwF4wf7/bot-chat.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''


user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/LZ2tYVh/icon.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>

'''


