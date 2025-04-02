playScreen = """
<PlayScreen>:
    name: "play"
    BoxLayout:
        orientation: 'vertical'
                
        MDTopAppBar:
            title: 'Play Screen'
            elevation: 3
            left_action_items: [["arrow-left", lambda x: root.go_back()]]
            
        BoxLayout:
            orientation: 'vertical'
            padding: [dp(10), dp(20), dp(10), dp(30)]
            pos_hint: {"center_x": 0.5, "center_y": 0.5}
            
            MDLabel:
                id: file_name_label
                halign: "center"
                size_hint_y: None
                height: dp(40)
                theme_text_color: "Primary"
                bold: True
            Widget
            MDSeparator
            Widget
            
            MDCarousel:
                id: carousel
                direction: "right"
                size_hint_y: None
                height: dp(550)
                halign: "center"
                pos_hint: {"center_y": 0.9}
                
            Widget:
                size_hint_y: 1
            MDSeparator
            Widget:
                size_hint_y: 1
                
            BoxLayout:
                spacing: "20dp"
                orientation: "horizontal"
                size_hint_y: None
                height: dp(70)
                halign: "center"
                pos_hint: {"center_x": 0.5}
                
                Widget:
                    size_hint_x: 0.1
                
                MDIconButton:
                    icon: "arrow-left"
                    on_release: root.carousel_previous()
                    size: dp(70), dp(70)
                    icon_size: "48sp"
    
                MDIconButton:
                    icon: "play"
                    on_release: root.play()
                    icon_size: "56sp"
                    size: dp(70), dp(70)
                    md_bg_color: app.theme_cls.primary_color
                    radius: [24, 24, 24, 24]
    
                MDIconButton:
                    icon: "arrow-right"
                    on_release: root.carousel_next()
                    icon_size: "48sp"
                    size: dp(70), dp(70)
                    
                Widget:
                    size_hint_x: 0.1
            Widget:
                size_hint_y: 1
            MDSeparator
            Widget:
                size_hint_y: 1
"""