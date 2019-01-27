#pragma once
#include "state.hpp"
#include "uct_node.hpp"

class MCTS{
    public:
        MCTS(float simulation_time, int num_actions, bool m_use_fruits);
        MCTS();
        virtual int get_action(State state);
        virtual UCTNode* get_root_node();
    protected:
		bool m_use_fruits;
        float m_simulation_time;
        int m_num_actions;
		UCTNode* m_root_node;
};