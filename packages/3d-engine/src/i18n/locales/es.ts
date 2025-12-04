/**
 * Spanish (es) Translations
 *
 * @module @neurectomy/3d-engine/i18n/locales/es
 * @agents @LINGUA @SCRIBE
 * @phase Phase 3 - Dimensional Forge
 */

import type { NamespaceTranslations } from "../types";

export const es: NamespaceTranslations = {
  common: {
    appName: "Neurectomy",
    loading: "Cargando...",
    error: "Error",
    success: "Éxito",
    warning: "Advertencia",
    info: "Información",

    actions: {
      save: "Guardar",
      cancel: "Cancelar",
      delete: "Eliminar",
      edit: "Editar",
      create: "Crear",
      update: "Actualizar",
      close: "Cerrar",
      confirm: "Confirmar",
      reset: "Restablecer",
      refresh: "Actualizar",
      export: "Exportar",
      import: "Importar",
      copy: "Copiar",
      paste: "Pegar",
      undo: "Deshacer",
      redo: "Rehacer",
      search: "Buscar",
      filter: "Filtrar",
      sort: "Ordenar",
      expand: "Expandir",
      collapse: "Contraer",
      selectAll: "Seleccionar Todo",
      deselectAll: "Deseleccionar Todo",
    },

    status: {
      active: "Activo",
      inactive: "Inactivo",
      pending: "Pendiente",
      completed: "Completado",
      failed: "Fallido",
      processing: "Procesando",
      ready: "Listo",
      idle: "Inactivo",
    },

    time: {
      now: "Ahora",
      today: "Hoy",
      yesterday: "Ayer",
      tomorrow: "Mañana",
      thisWeek: "Esta Semana",
      lastWeek: "Semana Pasada",
      thisMonth: "Este Mes",
      lastMonth: "Mes Pasado",
    },

    units: {
      seconds: {
        one: "{{count}} segundo",
        other: "{{count}} segundos",
      },
      minutes: {
        one: "{{count}} minuto",
        other: "{{count}} minutos",
      },
      hours: {
        one: "{{count}} hora",
        other: "{{count}} horas",
      },
      days: {
        one: "{{count}} día",
        other: "{{count}} días",
      },
      items: {
        one: "{{count}} elemento",
        other: "{{count}} elementos",
      },
    },
  },

  graph: {
    nodes: {
      title: "Nodos",
      node: "Nodo",
      create: "Crear Nodo",
      delete: "Eliminar Nodo",
      edit: "Editar Nodo",
      count: {
        one: "{{count}} nodo",
        other: "{{count}} nodos",
      },
      types: {
        standard: "Nodo Estándar",
        agent: "Nodo de Agente",
        data: "Nodo de Datos",
        process: "Nodo de Proceso",
        decision: "Nodo de Decisión",
        input: "Nodo de Entrada",
        output: "Nodo de Salida",
        cluster: "Clúster",
      },
      properties: {
        id: "ID del Nodo",
        label: "Etiqueta",
        type: "Tipo",
        position: "Posición",
        size: "Tamaño",
        color: "Color",
        metadata: "Metadatos",
      },
    },

    edges: {
      title: "Aristas",
      edge: "Arista",
      create: "Crear Arista",
      delete: "Eliminar Arista",
      edit: "Editar Arista",
      count: {
        one: "{{count}} arista",
        other: "{{count}} aristas",
      },
      types: {
        default: "Conexión Predeterminada",
        dataFlow: "Flujo de Datos",
        controlFlow: "Flujo de Control",
        dependency: "Dependencia",
        association: "Asociación",
        inheritance: "Herencia",
        composition: "Composición",
        bidirectional: "Bidireccional",
      },
      properties: {
        id: "ID de Arista",
        source: "Nodo Origen",
        target: "Nodo Destino",
        weight: "Peso",
        label: "Etiqueta",
        directed: "Dirigida",
      },
    },

    layout: {
      title: "Diseño",
      forceDirect: "Fuerza Dirigida",
      hierarchical: "Jerárquico",
      circular: "Circular",
      grid: "Cuadrícula",
      radial: "Radial",
      tree: "Árbol",
      dagre: "Dagre",
      apply: "Aplicar Diseño",
      autoLayout: "Diseño Automático",
    },

    operations: {
      addNode: "Agregar nodo",
      removeNode: "Eliminar nodo",
      connectNodes: "Conectar nodos",
      disconnectNodes: "Desconectar nodos",
      groupNodes: "Agrupar nodos",
      ungroupNodes: "Desagrupar nodos",
      mergeNodes: "Fusionar nodos",
      splitNode: "Dividir nodo",
    },
  },

  agent: {
    title: "Agente",
    name: "Nombre del Agente",
    type: "Tipo de Agente",
    status: "Estado",
    create: "Crear Agente",
    delete: "Eliminar Agente",
    configure: "Configurar Agente",

    types: {
      llm: "Agente LLM",
      tool: "Agente de Herramienta",
      router: "Agente Enrutador",
      orchestrator: "Orquestador",
      worker: "Agente Trabajador",
      supervisor: "Supervisor",
      retriever: "Agente Recuperador",
      custom: "Agente Personalizado",
    },

    statuses: {
      idle: "Inactivo",
      running: "Ejecutando",
      waiting: "Esperando",
      error: "Error",
      completed: "Completado",
      terminated: "Terminado",
    },

    actions: {
      start: "Iniciar Agente",
      stop: "Detener Agente",
      pause: "Pausar Agente",
      resume: "Reanudar Agente",
      restart: "Reiniciar Agente",
      clone: "Clonar Agente",
    },

    metrics: {
      title: "Métricas",
      executionTime: "Tiempo de Ejecución",
      tokensUsed: "Tokens Usados",
      requestCount: "Cantidad de Solicitudes",
      successRate: "Tasa de Éxito",
      errorRate: "Tasa de Error",
      avgResponseTime: "Tiempo de Respuesta Promedio",
    },

    workflow: {
      title: "Flujo de Trabajo",
      step: "Paso",
      steps: {
        one: "{{count}} paso",
        other: "{{count}} pasos",
      },
      input: "Entrada",
      output: "Salida",
      condition: "Condición",
      loop: "Bucle",
      parallel: "Paralelo",
      sequence: "Secuencia",
    },
  },

  visualization: {
    scene: {
      title: "Escena",
      camera: "Cámara",
      lighting: "Iluminación",
      background: "Fondo",
      grid: "Cuadrícula",
      axes: "Ejes",
    },

    camera: {
      perspective: "Perspectiva",
      orthographic: "Ortográfica",
      reset: "Restablecer Cámara",
      zoomIn: "Acercar",
      zoomOut: "Alejar",
      pan: "Desplazar",
      rotate: "Rotar",
      focus: "Enfocar en Selección",
      fitToView: "Ajustar a la Vista",
    },

    views: {
      title: "Vista",
      top: "Superior",
      bottom: "Inferior",
      front: "Frontal",
      back: "Posterior",
      left: "Izquierda",
      right: "Derecha",
      isometric: "Isométrica",
      custom: "Personalizada",
    },

    rendering: {
      title: "Renderizado",
      quality: "Calidad",
      low: "Baja",
      medium: "Media",
      high: "Alta",
      ultra: "Ultra",
      wireframe: "Malla",
      solid: "Sólido",
      textured: "Texturizado",
      shadows: "Sombras",
      antialiasing: "Antialiasing",
      bloom: "Brillo",
      ambientOcclusion: "Oclusión Ambiental",
    },

    animation: {
      title: "Animación",
      play: "Reproducir",
      pause: "Pausar",
      stop: "Detener",
      speed: "Velocidad",
      loop: "Bucle",
      reverse: "Invertir",
      frame: "Cuadro",
      timeline: "Línea de Tiempo",
    },

    selection: {
      title: "Selección",
      none: "Nada seleccionado",
      single: "{{name}} seleccionado",
      multiple: {
        one: "{{count}} elemento seleccionado",
        other: "{{count}} elementos seleccionados",
      },
      selectAll: "Seleccionar Todo",
      deselectAll: "Deseleccionar Todo",
      invertSelection: "Invertir Selección",
    },
  },

  accessibility: {
    title: "Accesibilidad",
    enabled: "Accesibilidad Habilitada",
    disabled: "Accesibilidad Deshabilitada",

    screenReader: {
      title: "Lector de Pantalla",
      enabled: "Modo Lector de Pantalla",
      announce: "Anunciar",
      description: "Descripción",
    },

    keyboard: {
      title: "Navegación por Teclado",
      shortcuts: "Atajos de Teclado",
      navigation: "Navegación",
      focusMode: "Modo de Enfoque",
    },

    visual: {
      title: "Configuración Visual",
      highContrast: "Alto Contraste",
      colorBlindMode: "Modo Daltonismo",
      normal: "Visión Normal",
      protanopia: "Protanopía (Ceguera al Rojo)",
      deuteranopia: "Deuteranopía (Ceguera al Verde)",
      tritanopia: "Tritanopía (Ceguera al Azul)",
      monochromacy: "Monocromacia (Escala de Grises)",
      reducedMotion: "Movimiento Reducido",
    },

    descriptions: {
      node: "Nodo {{type}} etiquetado {{label}} en posición {{x}}, {{y}}, {{z}}",
      edge: "Conexión de {{source}} a {{target}}",
      cluster: "Clúster que contiene {{count}} nodos",
      scene: "Escena con {{nodeCount}} nodos y {{edgeCount}} aristas",
    },
  },

  errors: {
    general: {
      unknown: "Ocurrió un error desconocido",
      network: "Error de red. Por favor verifica tu conexión.",
      timeout: "La solicitud expiró",
      notFound: "Recurso no encontrado",
      unauthorized: "No estás autorizado para realizar esta acción",
      forbidden: "Acceso prohibido",
      serverError: "Error del servidor. Por favor intenta más tarde.",
    },

    validation: {
      required: "{{field}} es requerido",
      invalid: "{{field}} es inválido",
      tooShort: "{{field}} debe tener al menos {{min}} caracteres",
      tooLong: "{{field}} debe tener como máximo {{max}} caracteres",
      outOfRange: "{{field}} debe estar entre {{min}} y {{max}}",
      invalidFormat: "El formato de {{field}} es inválido",
    },

    graph: {
      nodeNotFound: "Nodo no encontrado: {{id}}",
      edgeNotFound: "Arista no encontrada: {{id}}",
      duplicateNode: "Ya existe un nodo con este ID",
      duplicateEdge: "Esta conexión ya existe",
      selfLoop: "Los auto-bucles no están permitidos",
      cycleDetected: "Esta operación crearía un ciclo",
    },

    agent: {
      notFound: "Agente no encontrado: {{id}}",
      alreadyRunning: "El agente ya está ejecutándose",
      notRunning: "El agente no está ejecutándose",
      configInvalid: "La configuración del agente es inválida",
      executionFailed: "La ejecución del agente falló: {{reason}}",
    },
  },

  tooltips: {
    graph: {
      addNode: "Clic para agregar un nuevo nodo",
      deleteNode: "Eliminar este nodo y sus conexiones",
      connect: "Arrastrar para conectar a otro nodo",
      zoom: "Desplazar para acercar/alejar",
      pan: "Clic y arrastrar para desplazar la vista",
      select: "Clic para seleccionar, Ctrl+Clic para múltiples",
    },

    controls: {
      undo: "Deshacer última acción (Ctrl+Z)",
      redo: "Rehacer última acción (Ctrl+Y)",
      save: "Guardar cambios (Ctrl+S)",
      export: "Exportar datos del grafo",
      import: "Importar datos del grafo",
      settings: "Abrir configuración",
      help: "Abrir ayuda",
    },
  },

  notifications: {
    success: {
      saved: "Cambios guardados exitosamente",
      created: "{{item}} creado exitosamente",
      updated: "{{item}} actualizado exitosamente",
      deleted: "{{item}} eliminado exitosamente",
      exported: "Exportación completada exitosamente",
      imported: "Importación completada exitosamente",
    },

    info: {
      loading: "Cargando {{item}}...",
      processing: "Procesando {{item}}...",
      autoSave: "Guardando cambios automáticamente...",
    },

    warning: {
      unsavedChanges: "Tienes cambios sin guardar",
      confirmDelete: "¿Estás seguro de que quieres eliminar {{item}}?",
      irreversible: "Esta acción no se puede deshacer",
    },
  },

  settings: {
    title: "Configuración",

    categories: {
      general: "General",
      appearance: "Apariencia",
      accessibility: "Accesibilidad",
      performance: "Rendimiento",
      advanced: "Avanzado",
    },

    general: {
      language: "Idioma",
      autoSave: "Guardado Automático",
      autoSaveInterval: "Intervalo de Guardado Automático",
      notifications: "Notificaciones",
    },

    appearance: {
      theme: "Tema",
      darkMode: "Modo Oscuro",
      lightMode: "Modo Claro",
      systemDefault: "Predeterminado del Sistema",
      fontSize: "Tamaño de Fuente",
      uiScale: "Escala de Interfaz",
    },

    performance: {
      hardwareAcceleration: "Aceleración de Hardware",
      maxNodes: "Nodos Máximos",
      renderQuality: "Calidad de Renderizado",
      animationSpeed: "Velocidad de Animación",
    },
  },

  help: {
    title: "Ayuda",

    sections: {
      gettingStarted: "Comenzando",
      tutorials: "Tutoriales",
      shortcuts: "Atajos de Teclado",
      faq: "Preguntas Frecuentes",
      support: "Soporte",
    },

    shortcuts: {
      title: "Atajos de Teclado",
      navigation: {
        title: "Navegación",
        panLeft: "Desplazar Izquierda",
        panRight: "Desplazar Derecha",
        panUp: "Desplazar Arriba",
        panDown: "Desplazar Abajo",
        zoomIn: "Acercar",
        zoomOut: "Alejar",
        resetView: "Restablecer Vista",
      },
      editing: {
        title: "Edición",
        undo: "Deshacer",
        redo: "Rehacer",
        copy: "Copiar",
        paste: "Pegar",
        delete: "Eliminar",
        selectAll: "Seleccionar Todo",
      },
    },
  },
};
