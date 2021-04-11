# frecog
Módulo de reconhecimento facial do Pedro Valentin GTVD.

Foram "comentadas"/retiradas algumas features do código com o objetivo de adaptá-lo às necessidades do EVA, são elas:

* a visualização em tempo real
* a publicação MQTT
* a impressão de algumas informações no terminal

<<<<<<< HEAD
Ao ser executado, o módulo, passa a esperar por uma conexão tcp (localhost) na porta 3030. Após a conexão, o módulo faz as leituras da face e então retorna ao cliente tcp, a string da expressão (codificada em bytes). Então, a conexão é encerrada pelo módulo e este fica aguardando por uma nova conexão.
=======
Ao ser executado, o módulo, passa a esperar por uma conexão tcp (localhost) na porta 3030. Após a conexão, o módulo faz as leituras da face e então retorna ao cliente tcp, a string da expressão (codificada em bytes). Então, a conexão é encerrada pelo módulo e este fica aguardando por uma nova conexão.

>>>>>>> 6459d845518fbf8cdd7800870edb9ccddae70264
