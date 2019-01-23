// AIbscpp.cpp : Defines the entry point for the console application.
//
#include "state.hpp"
#include "mcts_agent.hpp"

#ifdef DRAW
#include "Draw.h"
#include <SFML/Graphics.hpp>
#endif
#include <iostream>
#include <fstream>
#include "HumanAgent.h"

void print_array(int** array) {
	for (int i = 0; i < 9; ++i) {
		for (int j = 0; j < 9; ++j) {
			std::cout << array[i][j] << "\t";
		}
		std::cout << std::endl;
	}
}

void save_states(std::vector<State> states, std::vector<std::vector<int>> moves, int winner, int field_size, int index){
	std::ofstream state_file;
	state_file.open ("state_" + std::to_string(index) + ".txt");
	state_file << "[" << std::endl;

	for (int state_idx = 0; state_idx < states.size(); ++state_idx){
		State state = states[state_idx];
		std::vector<coords> fruits = state.get_fruits();
		state_file << "\t{" << std::endl;
		state_file << "\t\t\"Height\": " << state.get_field_height() << std::endl;
		state_file << "\t\t\"Width\": " << state.get_field_width() << std::endl;
		state_file << "\t\t\"Winner\": " << winner << std::endl;
		state_file << "\t\t\"Fruits\": [";
		for(int fruit_idx = 0; fruit_idx < fruits.size(); fruit_idx++){
			state_file << "[" << fruits[fruit_idx].x << "," << fruits[fruit_idx].y << "]";
			if (fruit_idx < fruits.size() - 1){
				state_file << ",";
			}
		}
		state_file << "]" << std::endl;

		state_file << "\t\t\"Snakes\": [" << std::endl;
		for(int snake_idx = 0; snake_idx < state.get_snake_count(); snake_idx++){
			Snake snake = state.get_snake(snake_idx);
			std::vector<coords> body = snake.get_body();
			state_file << "\t\t\t{" << std::endl;
			state_file << "\t\t\t\t\"Move\": " << moves[state_idx][snake_idx] << std::endl;
			state_file << "\t\t\t\t\"Health\": " << snake.get_health() << std::endl;
			state_file << "\t\t\t\t\"Body\": [";
			for(int body_idx = 0; body_idx < body.size(); body_idx++){
				state_file << "[" << body[body_idx].x << "," << body[body_idx].y << "]";
				if (body_idx < body.size() - 1){
					state_file << ",";
				}
			}
			state_file << "]" << std::endl;
			state_file << "\t\t\t}";
			if (snake_idx < state.get_snake_count() - 1){
				state_file << ",";
			}
			state_file << std::endl;
		}
		state_file << "\t\t]" << std::endl;
		state_file << "\t}";
		if (state_idx < states.size() - 1){
			state_file << ",";
		}
		state_file << std::endl;
	}
	state_file << "]";
	state_file.close();
}

int main(int argc, char const *argv[])
{
	#ifdef DRAW
	Draw Window(7,7);
	#endif

	int field_size = 9;

	MCTSAgent agent1(0.3f, 3, 0, 100, true, true);
	MCTSAgent agent2(0.3f, 3, 1, 100, true, false);
	std::vector<int> wins;

	//srand((unsigned)time(0));
	for (int n = 0; n < 20; ++n) {
		State state(field_size, field_size, 2, 3);
		int winner = -2;
		int s1 = 0;
		int s2 = 0;
		
		std::vector<State> states;
		std::vector<std::vector<int>> moves;

		print_array(state.get_array());
		while (winner == -2) {
			#ifdef DRAW
			Window.clear();
			Window.draw_list(state.get_snake(0).get_body(), sf::Color::White);
			Window.draw_list(state.get_snake(1).get_body(), sf::Color::Green);
			Window.draw_list(state.get_fruits(), sf::Color::Red);
			Window.display();
			#endif
			
			std::cout << "before act" << std::endl;
			s1 = agent1.act(state);
			s2 = agent2.act(state);
			std::cout << "after act "<< std::endl;
			states.push_back(state);
			moves.push_back(std::vector<int>{s1,s2});
			
			
			state.move_snake(s1);
			winner = state.move_snake(s2);
			print_array(state.get_array());
			std::cout << "winner " << winner << std::endl;
			std::cout << std::endl;
		}
		wins.push_back(winner);
		save_states(states, moves, winner, field_size, n);
	}
	for (auto& win : wins) {
		std::cout << win << std::endl;
	}
	return 0;
}


