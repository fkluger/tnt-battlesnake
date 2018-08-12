import socket
from splinter import Browser
from selenium.webdriver.common.keys import Keys


class BattlesnakeService(object):
    def __init__(self, num_agents=2, battle_snake_opts=None, extra_snakes=[]):
        if not battle_snake_opts:
            self.opts = {
                "url": "http://localhost:3000",
                "board_width": 13,
                "board_height": 13,
                "turn_delay": 100,
                "max_food": 1,
            }
        else:
            self.opts = battle_snake_opts
        self.extra_snakes = extra_snakes
        self.num_agents = num_agents
        self.ip_address = socket.gethostbyname(socket.gethostname())
        self.browser = Browser()

    def fill_battlesnake_options(self):
        width_field = self.browser.find_by_id("game_form_width").first
        width_field.fill(self.opts["board_width"])
        height_field = self.browser.find_by_id("game_form_height").first
        height_field.fill(self.opts["board_height"])
        delay_field = self.browser.find_by_id("game_form_delay").first
        delay_field.fill(self.opts["turn_delay"])
        max_food_field = self.browser.find_by_id("game_form_max_food").first
        max_food_field.fill(self.opts["max_food"])

    def fill_snake_urls(self):
        for i in range(0, self.num_agents):
            snake_url_field = self.browser.find_by_id(
                "game_form_snakes_" + str(i) + "_url"
            )
            snake_url_field.fill("http://{}:808{}".format(self.ip_address, i))
        for idx, url in enumerate(self.extra_snakes):
            snake_url_field = self.browser.find_by_id(
                "game_form_snakes_" + str(self.num_agents + idx) + "_url"
            )
            snake_url_field.fill(url)

    def get_game_id(self):
        parts = self.browser.url.split("/")
        return parts[3:-1][0]

    def create_game(self):

        self.browser.visit(self.opts["url"])
        new_game_btn = self.browser.find_link_by_href("/new").first
        new_game_btn.click()
        self.fill_battlesnake_options()
        self.fill_snake_urls()
        create_game_btn = self.browser.find_by_css('button[type="submit"]').first
        create_game_btn.click()
        self.game_id = self.get_game_id()
        play_game_btn = self.browser.find_link_by_href(
            "/play/{}".format(self.game_id)
        ).first
        play_game_btn.click()
        print("Created game {}".format(self.game_id))

    def reset_game(self):
        body = self.browser.find_by_tag("body").first
        body.type("q")

    def start_game(self):
        self.reset_game()
        body = self.browser.find_by_tag("body").first
        body.type(Keys.ARROW_UP)

    def is_dead(self, name):
        dead_snakes = self.browser.find_by_css(
            ".scoreboard-dead-snake + .scoreboard-snake-info .scoreboard-name"
        )
        dead_snake_names = map(lambda s: s.text, dead_snakes)
        return name in dead_snake_names
