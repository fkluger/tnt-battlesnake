// AIbscpp.cpp : Defines the entry point for the console application.
//
#include "state.hpp"
#include "mcts_agent.hpp"

#ifdef DRAW
#include "Draw.h"
#include <SFML/Graphics.hpp>
#endif
#include <iostream>
#include "HumanAgent.h"

void print_array(int** array) {
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			std::cout << array[i][j] << "\t";
		}
		std::cout << std::endl;
	}
}

int main(int argc, char const *argv[])
{
	#ifdef DRAW
	Draw Window(7,7);
	#endif

	MCTSAgent agent1(0.3f, 3, 0, 100, true, true);
	MCTSAgent agent2(0.3f, 3, 1, 100, true, false);
	std::vector<int> wins;

	//srand((unsigned)time(0));
	for (int n = 0; n < 20; ++n) {
		State state(9, 9, 2, 3);
		int winner = -2;
		int s1 = 0;
		int s2 = 0;
		print_array(state.get_array());
		while (winner == -2) {
			#ifdef DRAW
			Window.clear();
			Window.draw_list(state.get_snake(0).get_body(), sf::Color::White);
			Window.draw_list(state.get_snake(1).get_body(), sf::Color::Green);
			Window.draw_list(state.get_fruits(), sf::Color::Red);
			Window.display();
			#endif
			s1 = agent1.act(state);
			s2 = agent2.act(state);
			state.move_snake(s1);
			winner = state.move_snake(s2);
			print_array(state.get_array());
			std::cout << std::endl;
		}
		wins.push_back(winner);
	}
	for (auto& win : wins) {
		std::cout << win << std::endl;
	}
	return 0;
}


