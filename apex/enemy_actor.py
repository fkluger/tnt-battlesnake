class EnemyActor:

    def __init__(self, actor):
        self.actor = actor

    def act(self, state):
        return self.actor.act(state)
