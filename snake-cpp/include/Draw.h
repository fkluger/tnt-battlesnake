#pragma once

#ifndef _DRAW_
#define _DRAW_

#ifdef DRAW
#include <SFML/Graphics.hpp>
#endif
#include <list>
#include "coords.hpp"

class Draw
{
public:
	Draw(int width, int height);
	~Draw();
	void draw_list(std::vector<coords>, sf::Color);
	void draw_block(coords, sf::Color);
	bool isOpen();
	int getDirection(int);
	void display();
	void close();
	void clear();

private:
	sf::RenderWindow* m_app;
	enum direction { UP, RIGHT, DOWN, LEFT, CLOSED };
};

#endif // !_DRAW_
