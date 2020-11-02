import PySimpleGUI as sg
import re

class TelaPrincipal:
    def __init__(self):
        #Layout
        sg.theme('NeutralBlue')
        layout = [
                    [sg.Text("Informe o texto que deseja avaliar", text_color='white', key='texto')],
                    [sg.Multiline(size=(40, 10))],
                    [sg.Button('Avaliar',  key='avaliar', size=(5,1)), sg.Button('Limpar', key='limpar', size=(5,1)), sg.Button('Sair', key='sair', size=(5,1))]
                ]
        #Janela
        janela = sg.Window("Detector de Fake News",layout, element_justification='c', icon='covid.jpg', no_titlebar='true')
        # Extrair os dados da janela
        self.button, self.values = janela.Read()
        
   
    def Iniciar(self):
        if self.button == 'sair':
            tela.fechar()
        if self.button == 'avaliar':
            textoavaliado = self.values['texto']
            print(textoavaliado)
            tela.validar()

    def fechar(self):
        sg.WIN_CLOSED

    def validar(self):
        if re.search('\\covid\\b', textoavaliado, re.IGNORECASE):
            print("A string tem covid")
        else:
            print ("A string não tem covid")

tela = TelaPrincipal()
tela.Iniciar()


            
    
