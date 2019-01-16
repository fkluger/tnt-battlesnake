#include "HumanAgent.h"
#ifdef DRAW
#include <SFML\Graphics.hpp>
#endif
#include <chrono>
#include <thread>

HumanAgent::HumanAgent()
{
}


HumanAgent::~HumanAgent()
{
}

int HumanAgent::act(State state)
{
	
	std::this_thread::sleep_for(std::chrono::milliseconds(300));
	int dir = 1;
	#ifdef DRAW
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
		dir = 0;
	}
	else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
		dir = 1;
	}
	else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
		dir = 2;
	}
	#endif

	return dir;
}
