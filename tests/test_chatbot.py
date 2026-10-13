from src.chatbot import BookBot


def test_chatbot_matches_greeting_and_genre_requests():
    bot = BookBot()
    assert bot.respond("hello")
    response = bot.respond("fantasy books")
    assert "fantasy" in response.lower() or "genre" in response.lower() or "book" in response.lower()


def test_chatbot_handles_unknown_queries():
    bot = BookBot()
    response = bot.respond("asdkjalskdj")
    assert "I’m not sure" in response or "help" in response.lower() or "understand" in response.lower()
