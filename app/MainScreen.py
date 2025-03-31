mainScreen = """
<MainScreen>:
    name: "main"
    BoxLayout:
        orientation: 'vertical'
                
        MDTopAppBar:
            title: 'Demo Application'
            elevation: 5
            
        BoxLayout:
            orientation: 'vertical'
            padding: [dp(30), dp(50), dp(30), dp(30)]
                
            BoxLayout:
                orientation: 'horizontal'
                spacing: dp(30)
                size_hint_y: None
                height: dp(70)
                padding: dp(20)
                
                BoxLayout:
                    size_hint_x: 0.5
                    size_hint_y: None
                    height: dp(80)
                    pos_hint: {"center_y": 0.5}
                    
                    MDRaisedButton:
                        text: "Load Images"
                        font_size: dp(22)
                        size_hint_x: 1
                        size_hint_y: 0.8
                        height: dp(70)
                        on_release: app.open_file_manager()
                        pos_hint: {"center_x": 0.5, "center_y": 0.5}
                    
                BoxLayout:
                    orientation: "vertical"
                    size_hint_x: 0.3
                    spacing: dp(10)
                    pos_hint: {"center_y": 0.5}
                    
                    MDTextField:
                        id: bpm_input
                        hint_text: "BPM"
                        input_filter: "int"
                        height: dp(50)
                        size_hint_x: 1
                        
                    BoxLayout:
                        orientation: 'horizontal'
                        spacing: dp(10)
                        size_hint_x: 1
                        size_hint_y: 1
                        height: dp(40)
                        
                        MDLabel:
                            text: "Swing"
                            valign: 'center'
                            size_hint_x: None
                            width: dp(45)
                            pos_hint: {"center_y": 0.5}
                        
                        MDCheckbox:
                            id: swing_checkbox
                            size_hint: None, None
                            size: dp(35), dp(35)
                            pos_hint: {"center_y": 0.5}
                        
            Widget:
                size_hint_y: 0.05
    
            MDScrollView:
                size_hint_y: 0.5
                MDList:
                    id: file_list
    
            Widget:
                size_hint_y: 0.05
                
            MDRaisedButton:
                id: play_button
                text: "Play"
                pos_hint: {"center_x": 0.5}
                size_hint_y: None
                height: dp(50)
                on_release: app.play()
                opacity: 0
                disabled: True
                
            Widget:
                size_hint_y: 0.05
"""