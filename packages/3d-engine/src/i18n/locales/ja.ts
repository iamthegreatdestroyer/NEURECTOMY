/**
 * Japanese (ja) Translations
 *
 * @module @neurectomy/3d-engine/i18n/locales/ja
 * @agents @LINGUA @SCRIBE
 * @phase Phase 3 - Dimensional Forge
 */

import type { NamespaceTranslations } from "../types";

export const ja: NamespaceTranslations = {
  common: {
    appName: "Neurectomy",
    loading: "読み込み中...",
    error: "エラー",
    success: "成功",
    warning: "警告",
    info: "情報",

    actions: {
      save: "保存",
      cancel: "キャンセル",
      delete: "削除",
      edit: "編集",
      create: "作成",
      update: "更新",
      close: "閉じる",
      confirm: "確認",
      reset: "リセット",
      refresh: "更新",
      export: "エクスポート",
      import: "インポート",
      copy: "コピー",
      paste: "貼り付け",
      undo: "元に戻す",
      redo: "やり直す",
      search: "検索",
      filter: "フィルター",
      sort: "並べ替え",
      expand: "展開",
      collapse: "折りたたむ",
      selectAll: "すべて選択",
      deselectAll: "すべて選択解除",
    },

    status: {
      active: "アクティブ",
      inactive: "非アクティブ",
      pending: "保留中",
      completed: "完了",
      failed: "失敗",
      processing: "処理中",
      ready: "準備完了",
      idle: "待機中",
    },

    time: {
      now: "今",
      today: "今日",
      yesterday: "昨日",
      tomorrow: "明日",
      thisWeek: "今週",
      lastWeek: "先週",
      thisMonth: "今月",
      lastMonth: "先月",
    },

    units: {
      seconds: {
        one: "{{count}}秒",
        other: "{{count}}秒",
      },
      minutes: {
        one: "{{count}}分",
        other: "{{count}}分",
      },
      hours: {
        one: "{{count}}時間",
        other: "{{count}}時間",
      },
      days: {
        one: "{{count}}日",
        other: "{{count}}日",
      },
      items: {
        one: "{{count}}件",
        other: "{{count}}件",
      },
    },
  },

  graph: {
    nodes: {
      title: "ノード",
      node: "ノード",
      create: "ノードを作成",
      delete: "ノードを削除",
      edit: "ノードを編集",
      count: {
        one: "{{count}}ノード",
        other: "{{count}}ノード",
      },
      types: {
        standard: "標準ノード",
        agent: "エージェントノード",
        data: "データノード",
        process: "プロセスノード",
        decision: "決定ノード",
        input: "入力ノード",
        output: "出力ノード",
        cluster: "クラスター",
      },
      properties: {
        id: "ノードID",
        label: "ラベル",
        type: "タイプ",
        position: "位置",
        size: "サイズ",
        color: "色",
        metadata: "メタデータ",
      },
    },

    edges: {
      title: "エッジ",
      edge: "エッジ",
      create: "エッジを作成",
      delete: "エッジを削除",
      edit: "エッジを編集",
      count: {
        one: "{{count}}エッジ",
        other: "{{count}}エッジ",
      },
      types: {
        default: "デフォルト接続",
        dataFlow: "データフロー",
        controlFlow: "制御フロー",
        dependency: "依存関係",
        association: "関連",
        inheritance: "継承",
        composition: "合成",
        bidirectional: "双方向",
      },
      properties: {
        id: "エッジID",
        source: "ソースノード",
        target: "ターゲットノード",
        weight: "重み",
        label: "ラベル",
        directed: "有向",
      },
    },

    layout: {
      title: "レイアウト",
      forceDirect: "フォースダイレクト",
      hierarchical: "階層的",
      circular: "円形",
      grid: "グリッド",
      radial: "放射状",
      tree: "ツリー",
      dagre: "Dagre",
      apply: "レイアウトを適用",
      autoLayout: "自動レイアウト",
    },

    operations: {
      addNode: "ノードを追加",
      removeNode: "ノードを削除",
      connectNodes: "ノードを接続",
      disconnectNodes: "接続を解除",
      groupNodes: "ノードをグループ化",
      ungroupNodes: "グループを解除",
      mergeNodes: "ノードをマージ",
      splitNode: "ノードを分割",
    },
  },

  agent: {
    title: "エージェント",
    name: "エージェント名",
    type: "エージェントタイプ",
    status: "ステータス",
    create: "エージェントを作成",
    delete: "エージェントを削除",
    configure: "エージェントを設定",

    types: {
      llm: "LLMエージェント",
      tool: "ツールエージェント",
      router: "ルーターエージェント",
      orchestrator: "オーケストレーター",
      worker: "ワーカーエージェント",
      supervisor: "スーパーバイザー",
      retriever: "リトリーバーエージェント",
      custom: "カスタムエージェント",
    },

    statuses: {
      idle: "待機中",
      running: "実行中",
      waiting: "待機中",
      error: "エラー",
      completed: "完了",
      terminated: "終了",
    },

    actions: {
      start: "エージェントを開始",
      stop: "エージェントを停止",
      pause: "エージェントを一時停止",
      resume: "エージェントを再開",
      restart: "エージェントを再起動",
      clone: "エージェントを複製",
    },

    metrics: {
      title: "メトリクス",
      executionTime: "実行時間",
      tokensUsed: "使用トークン",
      requestCount: "リクエスト数",
      successRate: "成功率",
      errorRate: "エラー率",
      avgResponseTime: "平均応答時間",
    },

    workflow: {
      title: "ワークフロー",
      step: "ステップ",
      steps: {
        one: "{{count}}ステップ",
        other: "{{count}}ステップ",
      },
      input: "入力",
      output: "出力",
      condition: "条件",
      loop: "ループ",
      parallel: "並列",
      sequence: "シーケンス",
    },
  },

  visualization: {
    scene: {
      title: "シーン",
      camera: "カメラ",
      lighting: "ライティング",
      background: "背景",
      grid: "グリッド",
      axes: "軸",
    },

    camera: {
      perspective: "透視投影",
      orthographic: "平行投影",
      reset: "カメラをリセット",
      zoomIn: "ズームイン",
      zoomOut: "ズームアウト",
      pan: "パン",
      rotate: "回転",
      focus: "選択にフォーカス",
      fitToView: "ビューに合わせる",
    },

    views: {
      title: "ビュー",
      top: "上",
      bottom: "下",
      front: "前",
      back: "後",
      left: "左",
      right: "右",
      isometric: "アイソメトリック",
      custom: "カスタム",
    },

    rendering: {
      title: "レンダリング",
      quality: "品質",
      low: "低",
      medium: "中",
      high: "高",
      ultra: "ウルトラ",
      wireframe: "ワイヤーフレーム",
      solid: "ソリッド",
      textured: "テクスチャ",
      shadows: "影",
      antialiasing: "アンチエイリアス",
      bloom: "ブルーム",
      ambientOcclusion: "アンビエントオクルージョン",
    },

    animation: {
      title: "アニメーション",
      play: "再生",
      pause: "一時停止",
      stop: "停止",
      speed: "速度",
      loop: "ループ",
      reverse: "逆再生",
      frame: "フレーム",
      timeline: "タイムライン",
    },

    selection: {
      title: "選択",
      none: "何も選択されていません",
      single: "{{name}}を選択中",
      multiple: {
        one: "{{count}}件を選択中",
        other: "{{count}}件を選択中",
      },
      selectAll: "すべて選択",
      deselectAll: "すべて選択解除",
      invertSelection: "選択を反転",
    },
  },

  accessibility: {
    title: "アクセシビリティ",
    enabled: "アクセシビリティ有効",
    disabled: "アクセシビリティ無効",

    screenReader: {
      title: "スクリーンリーダー",
      enabled: "スクリーンリーダーモード",
      announce: "アナウンス",
      description: "説明",
    },

    keyboard: {
      title: "キーボードナビゲーション",
      shortcuts: "キーボードショートカット",
      navigation: "ナビゲーション",
      focusMode: "フォーカスモード",
    },

    visual: {
      title: "ビジュアル設定",
      highContrast: "ハイコントラスト",
      colorBlindMode: "色覚モード",
      normal: "通常視力",
      protanopia: "赤色盲",
      deuteranopia: "緑色盲",
      tritanopia: "青色盲",
      monochromacy: "モノクロ",
      reducedMotion: "モーション削減",
    },

    descriptions: {
      node: "{{label}}というラベルの{{type}}ノード、位置{{x}}, {{y}}, {{z}}",
      edge: "{{source}}から{{target}}への接続",
      cluster: "{{count}}ノードを含むクラスター",
      scene: "{{nodeCount}}ノードと{{edgeCount}}エッジを持つシーン",
    },
  },

  errors: {
    general: {
      unknown: "不明なエラーが発生しました",
      network: "ネットワークエラー。接続を確認してください。",
      timeout: "リクエストがタイムアウトしました",
      notFound: "リソースが見つかりません",
      unauthorized: "この操作を行う権限がありません",
      forbidden: "アクセスが禁止されています",
      serverError: "サーバーエラー。後でもう一度お試しください。",
    },

    validation: {
      required: "{{field}}は必須です",
      invalid: "{{field}}が無効です",
      tooShort: "{{field}}は{{min}}文字以上必要です",
      tooLong: "{{field}}は{{max}}文字以下にしてください",
      outOfRange: "{{field}}は{{min}}から{{max}}の間である必要があります",
      invalidFormat: "{{field}}のフォーマットが無効です",
    },

    graph: {
      nodeNotFound: "ノードが見つかりません: {{id}}",
      edgeNotFound: "エッジが見つかりません: {{id}}",
      duplicateNode: "このIDのノードは既に存在します",
      duplicateEdge: "この接続は既に存在します",
      selfLoop: "自己ループは許可されていません",
      cycleDetected: "この操作は循環を作成します",
    },

    agent: {
      notFound: "エージェントが見つかりません: {{id}}",
      alreadyRunning: "エージェントは既に実行中です",
      notRunning: "エージェントは実行されていません",
      configInvalid: "エージェントの設定が無効です",
      executionFailed: "エージェントの実行に失敗しました: {{reason}}",
    },
  },

  tooltips: {
    graph: {
      addNode: "クリックして新しいノードを追加",
      deleteNode: "このノードとその接続を削除",
      connect: "ドラッグして別のノードに接続",
      zoom: "スクロールでズームイン/アウト",
      pan: "クリック＆ドラッグでビューを移動",
      select: "クリックで選択、Ctrl+クリックで複数選択",
    },

    controls: {
      undo: "元に戻す (Ctrl+Z)",
      redo: "やり直す (Ctrl+Y)",
      save: "変更を保存 (Ctrl+S)",
      export: "グラフデータをエクスポート",
      import: "グラフデータをインポート",
      settings: "設定を開く",
      help: "ヘルプを開く",
    },
  },

  notifications: {
    success: {
      saved: "変更が正常に保存されました",
      created: "{{item}}が正常に作成されました",
      updated: "{{item}}が正常に更新されました",
      deleted: "{{item}}が正常に削除されました",
      exported: "エクスポートが正常に完了しました",
      imported: "インポートが正常に完了しました",
    },

    info: {
      loading: "{{item}}を読み込み中...",
      processing: "{{item}}を処理中...",
      autoSave: "変更を自動保存中...",
    },

    warning: {
      unsavedChanges: "保存されていない変更があります",
      confirmDelete: "{{item}}を削除してもよろしいですか？",
      irreversible: "この操作は元に戻せません",
    },
  },

  settings: {
    title: "設定",

    categories: {
      general: "一般",
      appearance: "外観",
      accessibility: "アクセシビリティ",
      performance: "パフォーマンス",
      advanced: "詳細設定",
    },

    general: {
      language: "言語",
      autoSave: "自動保存",
      autoSaveInterval: "自動保存間隔",
      notifications: "通知",
    },

    appearance: {
      theme: "テーマ",
      darkMode: "ダークモード",
      lightMode: "ライトモード",
      systemDefault: "システム設定に従う",
      fontSize: "フォントサイズ",
      uiScale: "UIスケール",
    },

    performance: {
      hardwareAcceleration: "ハードウェアアクセラレーション",
      maxNodes: "最大ノード数",
      renderQuality: "レンダリング品質",
      animationSpeed: "アニメーション速度",
    },
  },

  help: {
    title: "ヘルプ",

    sections: {
      gettingStarted: "始めましょう",
      tutorials: "チュートリアル",
      shortcuts: "キーボードショートカット",
      faq: "よくある質問",
      support: "サポート",
    },

    shortcuts: {
      title: "キーボードショートカット",
      navigation: {
        title: "ナビゲーション",
        panLeft: "左に移動",
        panRight: "右に移動",
        panUp: "上に移動",
        panDown: "下に移動",
        zoomIn: "ズームイン",
        zoomOut: "ズームアウト",
        resetView: "ビューをリセット",
      },
      editing: {
        title: "編集",
        undo: "元に戻す",
        redo: "やり直す",
        copy: "コピー",
        paste: "貼り付け",
        delete: "削除",
        selectAll: "すべて選択",
      },
    },
  },
};
