#pragma once
#include <vector>
#include "snake.hpp"
#include "coords.hpp"

class State{
    public:
        State(int width, int height, int num_snakes, int num_fruits);
		State(const State &old_state);
		State() {};
        int move_snake(int action);
		Snake get_snake(int idx);
		std::vector<coords> get_fruits();
        int get_current_player();
        void set_current_player(int idx);
        int** get_array();
		void set_health(int health);
    private:
        int move_snakes(std::vector<int> actions);
        void collision_check();
        bool ate_fruit(Snake snake);
        void place_fruits_or_snakes(int num, bool fruits);
        bool is_available(coords field);
        void update_snakes_alive();

        int m_width;
        int m_height;
        int m_current_snake_idx;
        int m_alive_idx;
        int m_num_snakes;
        int m_num_fruits;
        
        std::vector<coords> m_fruits;
        std::vector<Snake> m_snakes;
        std::vector<Snake*> m_snakes_alive;
};