import gc, sys
from modules.assistant import ModelName
from modules.assistant import Assistant
from tqdm import tqdm


exam = {
    "questions" : [
        ("¿De qué maneras podemos aportar documentación a un expediente de MyTAO?\n"),
        ("Indique las formas de generar un encargo:\n"),
        ("¿De las siguientes opciones, dónde se visualizan las anotaciones " 
        "que aún no han sido "
        "validadas por una unidad organizativa?\n"),
        ("Cuáles son los posibles estados de una notificación electrónica que "
        "se ha puesto a "
        "disposición para un obligado legal:\n"),
        ("¿Qué utilidad permite incluir información interna visible para "
        "el equipo dentro de una "
        "anotación, expediente o encargo?\n"),
        ("¿Cuál es el objetivo de la bandeja SIR?\n"),
        ("¿Qué ocurre si subimos un documento sin firma a un expediente?\n"),
        ("¿Qué representa el estado “PA” en un encargo?\n"),
        ("¿Cuál es la ventaja del uso de marcadores en documentos de Word?\n"),
        ("¿Qué significa AAA y cual es un ejemplo de AAA en la plataforma?\n"),
        ("¿Cuál de los siguientes es un canal de la bandeja de encargos?\n"),
        ("¿Qué sucede si una anotación supera los 30 días en el canal de "
        "«Mis Anotaciones (último mes)»?\n"),
        ("¿Dónde se realiza el cambio de clasificación (catalogación) de una anotación?\n"),
        ("Para las personas físicas, que no han dado su consentimiento para relacionarse "
        "electrónicamente con el Ayuntamiento, cuando les emitimos una notificación "
        "fehaciente, ¿Se pone a disposición igualmente la notificación electrónica?\n"),
        ("¿Qué debe incluir un documento electrónico impreso para ser considerado válido?\n"),
        ("¿Qué tipo de canal requiere gestión manual por parte del usuario?\n"),
        ("¿Qué funcionalidad permite firmar varios documentos a la vez desde "
        "el Portafirmas?\n"),
        ("¿Qué visualiza el número entre paréntesis junto al nombre de una bandeja "
        "de usuario?\n"),
        ("En relación a las personas que tienen obligación de relacionarse "
        "electrónicamente con nuestro Ayuntamiento, ¿qué significan las siglas EV y "
        "OL que aparece en el expediente?\n"),
        ("¿Cuál es la URL de la Sede Electrónica del Ayuntamiento de Benalmádena?\n")
    ],
    "options" : [
        (
            "1. Desde el propio expediente: mediante actuaciones que generan documentos como es el "
            "caso del “informe técnico” o bien aportando los documentos por el botón de “ "
            "Documentos”\n"
            "2. A través de un encargo recibido con la acción de “Relacionar con el expediente los "
            "documentos anexados al encargo”\n"
            "3. A través de una anotación, ya sea al abrir el expediente desde la propia anotación o bien "
            "al incorporar la anotación mediante la actuación de “recepción de documentación "
            "complementaria”\n"
            "4. Todas las respuestas son correctas\n"
        ),
        (
            "1. Desde un expediente\n"
            "2. Desde otro encargo\n"
            "3. Desde una anotación\n"
            "4. Las opciones 1 y 2 son correctas\n"
        ),
        (
            "1. Bandeja “Documentos”\n"
            "2. Canal “Anotaciones de mi UO pendientes de validar\n"
            "3. Canal “Mis encargos pendientes”\n"
            "4. Registro de salida\n"
        ),
        (
            "1. Notificada únicamente\n"
            "2. Notificada (Aceptación), Notificada (Rechazo), Notificada (Expiración)\n"
            "3. Pendiente de firma y firmada\n"
            "4. Todas las opciones son correctas\n"
        ),
        (
            "1. Canal de incidencias\n"
            "2. SIR\n"
            "3. Notas\n"
            "4. Marcas de revisión\n"
        ),
        (
            "1. Firmar electrónicamente documentos\n"
            "2. Controlar los registros de entrada\n"
            "3. Seguir el estado de los registros enviados por SIR\n"
            "4. Validar usuarios externos\n"
        ),
        (
            "1. Se rechaza automáticamente\n"
            "2. No se incluye en el índice del foliado electrónico del mismo\n"
            "3. Se firma por defecto\n"
            "4. No se puede visualizar\n"
        ),
        (
            "1. Pospuesto Aceptado\n"
            "2. Pendiente de Aceptar\n"
            "3. Para Auditar\n"
            "4. Pendiente Actuación\n"
        ),
        (
            "1. Validan automáticamente los documentos\n"
            "2. Facilitan la exportación a Excel\n"
            "3. Sustituyen dinámicamente campos por datos del expediente\n"
            "4. Eluden la firma electrónica\n"
        ),
        (
            "1. Área Administrativa Automatizada. Ejemplo: Informática,\n"
            "2. Actuación Administrativa Automatizada. Ejemplo: La notificación de resolución o "
            "la descarga inmediata del volante de empadronamiento desde la Sede Electrónica\n"
            "3. Aplicación Avanzada de Anotaciones. Ejemplo: Anotaciones pendientes de validar\n"
            "4. Archivo de Acciones Automáticas. Ejemplo. SIGA\n"
        ),
        (
            "1. Documentos firmados\n"
            "2. Expedientes relacionados\n"
            "3. Mis encargos pendientes de tramita\n"
            "4. Actuaciones completadas\n"
        ),
        (
            "1. Se elimina del sistema\n"
            "2. Se archiva automáticamente\n"
            "3. Deja de visualizarse en el canal\n"
            "4. Cambia a estado “cerrada”\n"
        ),
        (
            "1. En la pestaña “Documentos”\n"
            "2. En la sección “Contenido” del detalle de la anotación\n"
            "3. En el panel de usuario\n"
            "4. No se puede re-catalogar una anotación\n"
        ),
        (
            "1. No, ya que no están obligados\n"
            "2. Sí, pero no están obligados a entrar en la Sede Electrónica para verla\n"
            "3. Sí, pero una vez emitida ya están obligados entrar en la Sede Electrónica para verla\n"
            "4. Depende de si están o no empadronados en el municipio\n"
        ),
        (
            "1. Firma manuscrita\n"
            "2. Código Seguro Validación (CSV)\n"
            "3. Fecha del expediente\n"
            "4. Nombre del funcionario\n"
        ),
        (
            "1. “Mis encargos pendientes de tramitar”\n"
            "2. “Encargos solicitados y rechazados”\n"
            "3. “Anotaciones pendientes de validar”\n"
            "4. “Documentos pendientes de mi firma o visado”\n"
        ),
        (
            "1. Visado\n"
            "2. Firma delegada\n"
            "3. Firma masiva\n"
            "4. Encargo múltiple\n"
        ),
        (
            "1. Número total de expediente\n"
            "2. Número de firmas digitales\n"
            "3. Número de ítems pendientes en sus canales\n"
            "4. Número de usuarios conectados\n"
        ),
        (
            "1. EV indica Electrónica Válida y OL indica Obligado Literal\n"
            "2. EV indica Electrónico Voluntario y OL indica Obligado Legal\n"
            "3. EV indica Es Voluntario y OL indica Organismo Legal\n"
            "4. Esas siglas son incorrectas\n"
        ),
        (
            "1. http://benalmadena.es/sede\n"
            "2. http://www.benalmadena.es/\n"
            "3. https://sede.benalmadena.es/\n"
            "4. sede@benalmadena.es\n"
        )
    ],
    "correct" : [
        "4", "4", "2", "2", "3", "3", "2", "2", "3", 
        "2", "3", "3", "2", "2", "2", "2", "3", "3", 
        "2", "3"
    ]
}


if __name__ == "__main__":
    list_k = [1, 2, 3, 5, 7]
    models = [ModelName.GRANITE_NANO]

    for model in models:
        for k in list_k:
            assistant = Assistant(model, k=k, temperature=0.01)
            assistant.benchmark(exam)
            del assistant; gc.collect()
