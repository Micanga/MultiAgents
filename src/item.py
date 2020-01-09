import position


class item:

    def __init__(self, x,y,level, index):
        self.position = position.position(int(x), int(y))
        self.level = float(level)
        self.index = index
        self.loaded = False
        self.agents_load_item = list()

        self.already_seen = False

    def is_agent_in_loaded_list(self, new_agent):
        for agent in self.agents_load_item:
            if new_agent.intelligent_agent:
                if agent.intelligent_agent:
                    return True
            else:
                if agent.equals(new_agent):
                    return True
        return False

    def remove_agent(self,agent_x , agent_y):
        for agent_load in self.agents_load_item:
            (al_x , al_y) = agent_load.get_position()
            if al_x == agent_x and al_y == agent_y:
                self.agents_load_item.remove(agent_load)
                return

    def get_position(self):
        return (self.position.x, self.position.y)

    def set_position (self , x,y):
        self.position.x = x
        self.position.y = y

    def copy(self):

        (x, y) = self.get_position()

        copy_item = item(x, y, self.level, self.index)

        copy_item.loaded = self.loaded

        copy_item.already_seen = self.already_seen


        ca_list = list()
        for a in self.agents_load_item:
            ca =  a.copy()
            ca_list.append(ca)

        copy_item.agents_load_item = ca_list

        return copy_item
        
    def equals(self, other_item):
        (other_x, other_y) = other_item.get_position()
        (x, y) = self.get_position()

        return (other_x == x) and (other_y == y) and\
               other_item.loaded == self.loaded and \
               other_item.level == self.level and \
               other_item.index == self.index
  
