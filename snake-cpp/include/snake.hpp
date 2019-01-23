#pragma once
#include "coords.hpp"
#include <vector>

enum direction { left=0, up=1, right=2, down=3};

class Snake{
     private:
        direction get_direction(int action);
        coords get_next_head(direction dir);
		
        bool m_alive;
        int m_health;
        direction m_head_direction;
        std::vector<coords> m_body;
        int m_idx;
        int m_max_length;
        int m_maxhealth;

    public:
        Snake(coords head, int idx);
        void move_tail(bool ate_fruit);
        coords get_head();
		int get_length();
		void set_health(int health);
        void die();
        bool is_dead();
        std::vector<coords> get_body();
        void dead();
		int get_health();
        int get_idx();
        void move_head(int action);

        bool operator!=(const Snake& a) const{
            return (m_body != a.m_body);
        }
};
