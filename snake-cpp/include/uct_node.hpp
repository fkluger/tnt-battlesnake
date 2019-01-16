#include "state.hpp"
#pragma once

class UCTNode{
    public:
        UCTNode(State state, int active_player, int action, int num_actions, int winner = -2, UCTNode* parent = nullptr);
		UCTNode(State state, int active_player, int action, int num_actions, std::vector<UCTNode*> children);
		UCTNode();
		~UCTNode();
        void set_visits(float visits);
        void set_value(float value);
		int get_winner();
        State get_state();
        UCTNode* get_child(int best_action);
        void set_child(int best_action, UCTNode* node);
        UCTNode* get_parent();
        UCTNode* select();
        int get_active_player();
        void backup(int winner, int fruits_eaten);
        void expand();
        std::vector<float> res();
        float get_visits();
		float get_value();
    private:
        
        std::vector<float> child_uct();
        bool is_leaf();
        float uct(float child_value, float child_visits);
		void set_parent(UCTNode* parent);

        bool m_isleaf;
        int m_active_player;
        int m_num_actions;
        int m_action;
		float m_value;
		float m_visits;
		int m_winner;
        std::vector<UCTNode*> m_children;
        UCTNode* m_parent;
        State m_state;
};
    