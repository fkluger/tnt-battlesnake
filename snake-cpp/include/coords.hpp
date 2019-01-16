#pragma once

class coords{
    public:
        int x;
        int y;

        coords(int x, int y){
            this->x = x;
            this->y = y;
        }

        bool operator==(const coords& a) const{
            return (x == a.x && y == a.y);
        }

        bool operator!=(const coords& a) const{
            return (x != a.x || y != a.y);
        }
};