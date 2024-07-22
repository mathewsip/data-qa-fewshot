few_shots_ag = [
    {'input': "What is my most resourced project?", 'query': 'SELECT project_id, SUM(hours_resourced) AS total_hours_resourced FROM Resourced_Data GROUP BY project_id ORDER BY total_hours_resourced DESC LIMIT 1;'},
    {'input': "What is my most resourced task?", 'query': 'SELECT task_name, SUM(hours_resourced) AS total_hours_resourced\nFROM Resourced_Data\nGROUP BY task_name\nORDER BY total_hours_resourced DESC\nLIMIT 1;'},
    {'input': "What is my most resourced role?", 'query': 'SELECT resourced_role, SUM(hours_resourced) AS total_hours_resourced\nFROM Resourced_Data\nGROUP BY resourced_role\nORDER BY total_hours_resourced DESC\nLIMIT 1;'},
] 