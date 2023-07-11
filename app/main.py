import cv2
import matrix_calculations
import data_send
import pygame as pg
import pandas as pd
import time

calculations = matrix_calculations.AttentioRecognition()
data = data_send.DataSending()

class Game:
    def __init__(self):
        """ Initialize the game window, etc. """
        pg.init()
        self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
        pg.display.flip()
        self.clock = pg.time.Clock()
        self.running = True
        self.frame_count = 0

        self.scale = 0

        #data.clear()

    def intialize(self):
        self.logo = pg.image.load("logo.png")
        # tela branca para mostrar o logo
        self.screen.fill((255, 255, 255))
        # redimensionar o logo
        logo = pg.transform.scale(self.logo, (self.screen.get_width(), self.screen.get_height()))
        # desenhar o logo
        self.screen.blit(self.logo, (0, 0))
        pg.display.flip()
        pg.time.wait(3000)

        self.logo = pg.image.load("desenho.png")

        # redimensionar para 200x200
        self.logo = pg.transform.scale(self.logo, (200, 200))

    def run(self):
        ''' Run the game loop.'''
        while self.running:
            self.frame_count += 1
            self.clock.tick(60)
            pg.display.flip()
            
            frame = calculations.run()

            # Transformar o fram em uma imagem do pygame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = pg.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")

            # redimensionar a imagem para o tamanho da tela
            frame = pg.transform.scale(frame, (self.screen.get_width(), self.screen.get_height()))

            # Desenhar a imagem na tela
            self.screen.blit(frame, (0, 0))

            # colocar a logo no canto da tela
            self.screen.blit(self.logo, (0, self.screen.get_height() - self.logo.get_height()))

            # printar a media de atencao na tela
            calculations.print_medias(self.screen)

            if self.frame_count % 10 == 0:
                media, media5, atention, alunos = calculations.retornar_media()
                if len(atention) > 0:
                    print('enviando dados')
                    columns = [f"id{i}" for i in range(len(atention))]
                    columns.insert(0, "alunos")
                    columns.insert(0, "media")
                    columns.insert(0, "tempo")
                    atention.insert(0, alunos)
                    atention.insert(0, media5)
                    # adicionar tempo no formato dd/mm/yyyy hh:mm:ss
                    atention.insert(0, time.strftime("%d/%m/%Y %H:%M:%S", time.localtime()))
                    pd.DataFrame([atention], columns=columns).to_csv("data.csv", index=False)
                    data.enviar_dados(media5)
                calculations.update_graf()

            calculations.aviso_baixa_atencao(self.screen)
            calculations.print_graf(self.screen, self.scale)

            # verificar se apertou esc
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.running = False
                    # se apertar l limpa a planilha
                    if event.key == pg.K_l:
                        data.clear()
                    if event.key == pg.K_g:
                        self.scale = not self.scale
                        calculations.update_graf()
                    

        pg.quit()


if __name__ == '__main__':
    ''' Create an instance of the Game class and start the game loop.'''
    game = Game()
    game.intialize()
    game.run()
    calculations.cap.release()
    cv2.destroyAllWindows()
