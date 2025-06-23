class DataService:
    def __init__(self, config_manager, data_collector):
        self.config_manager = config_manager
        self.data_collector = data_collector
    
    def get_collected_data_info(self):
        """Retorna informações sobre os dados coletados"""
        return self.data_collector.get_collected_data_info()
    
    def get_previous_configs(self):
        """Retorna configurações anteriores"""
        configs = self.config_manager.get_previous_configs()
        return {
            'configs': configs,
            'total_configs': len(configs)
        }
