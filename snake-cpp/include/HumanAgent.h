#pragma once
#include "Agent.h"
class HumanAgent : Agent
{
public:
	HumanAgent();
	~HumanAgent();
	int act(State state) override;
};

