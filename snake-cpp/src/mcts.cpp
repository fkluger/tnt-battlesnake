#include "mcts.hpp"
#include <chrono>
#include <iostream>
#include <cstdlib> 
#include <numeric>
#include <algorithm>

MCTS::MCTS(float simulation_time, int num_actions, bool use_fruits):
    m_simulation_time(simulation_time), m_num_actions(num_actions), m_use_fruits(use_fruits){   
}

MCTS::MCTS(){}

int MCTS::get_action(State state){
    
    srand((unsigned)time(0));
    m_root_node = new UCTNode(
        state,
        state.get_current_player(),
        0,
        m_num_actions,
		-2,
        nullptr
    );
	auto start_time = std::chrono::steady_clock::now();
    auto end_time = std::chrono::steady_clock::now();
	while(std::chrono::duration_cast<std::chrono::milliseconds>
        (end_time - start_time).count() < m_simulation_time*1000){
		end_time = std::chrono::steady_clock::now();
        UCTNode* leaf_node = m_root_node->select();
		int winner = leaf_node->get_winner();
        if (winner != -2){
            leaf_node->backup(winner, 0);
            continue;
        } else {
            State discover_state = leaf_node->get_state();
			int snake_length = discover_state.get_snake(discover_state.get_current_player()).get_length();
			while (true) {
				winner = discover_state.move_snake(rand() % m_num_actions);
				if (winner != -2) {
					break;
				}
			}
			int snake_length_dif = snake_length - discover_state.get_snake(discover_state.get_current_player()).get_length();
            leaf_node->expand();
            leaf_node->backup(winner, (m_use_fruits) ? snake_length_dif : 0);
        }
    }
    std::vector<float> v = m_root_node->res();
    auto result = std::max_element(v.begin(), v.end());
    int best_action = std::distance(v.begin(), result);

    std::cout << m_root_node->get_visits() << std::endl;
	delete m_root_node;
    return best_action;
}

UCTNode* MCTS::get_root_node(){
    return m_root_node;
}