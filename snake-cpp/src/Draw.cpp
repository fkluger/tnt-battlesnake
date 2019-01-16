#include "Draw.h"

Draw::Draw(int width, int height)
{
	int window_height = height;
	int window_width = width;
	m_app = new sf::RenderWindow(sf::VideoMode(window_width * 25, window_height * 25), "Snake 1.0 par Mathieu Allaire", sf::Style::Close);
	m_app->setFramerateLimit(12);
}

void Draw::draw_list(std::vector<coords> elements, sf::Color Color)
{
	for (std::vector<coords>::iterator it = elements.begin(); it != elements.end(); it++)
	{
		draw_block(*it, Color);
	}
}

void Draw::draw_block(coords element, sf::Color Color)
{
	float x = (element.x - 1) * 25;
	float y = (element.y - 1) * 25;

	sf::RectangleShape block = sf::RectangleShape(sf::Vector2f(20, 20));
	block.setFillColor(Color);
	block.setPosition(x, y);

	m_app->draw(block);
}

int Draw::getDirection(int cur_direction) {
	sf::Event Event;

	while (m_app->pollEvent(Event))
	{
		if (Event.type == sf::Event::Closed)
		{
			m_app->close();
		}
		if ((Event.type == sf::Event::KeyPressed) && (Event.key.code == sf::Keyboard::Return))
		{
			return -2;
		}
		// Determine the direction of the snake
		if ((Event.type == sf::Event::KeyPressed) && (Event.key.code == sf::Keyboard::Right))
		{
			return RIGHT;
		}
		else if ((Event.type == sf::Event::KeyPressed) && (Event.key.code == sf::Keyboard::Left))
		{
			return LEFT;
		}
		else if ((Event.type == sf::Event::KeyPressed) && (Event.key.code == sf::Keyboard::Up))
		{
			return UP;
		}
		else if ((Event.type == sf::Event::KeyPressed) && (Event.key.code == sf::Keyboard::Down))
		{
			return DOWN;
		}
	}
	return -1;
}

bool Draw::isOpen() {
	return m_app->isOpen();
}

void Draw::display() {
	m_app->display();
}

void Draw::clear() {
	m_app->clear();
}

void Draw::close() {
	m_app->close();
}

Draw::~Draw()
{
	delete m_app;
}
