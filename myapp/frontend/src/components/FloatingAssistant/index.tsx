import React, { useCallback, useEffect, useRef, useState } from 'react';
import AssistantPanel from './AssistantPanel';
import { useAssistant } from './useAssistant';
import tubiaoImg from '../../images/tubiao.png';
import './style.css';

const COLLAPSED_KEY = 'floating_assistant_collapsed';
const PANEL_WIDTH = 384;
const PANEL_HEIGHT = 540;
const BUBBLE_SIZE = 64;

const FloatingAssistant: React.FC = () => {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    return localStorage.getItem(COLLAPSED_KEY) !== '0';
  });
  const [position, setPosition] = useState<{ x: number; y: number } | null>(null);
  const {
    messages,
    sessions,
    currentSessionId,
    loading,
    initialized,
    send,
    stop,
    clearHistory,
    regenerate,
    switchSession,
    newSession,
  } = useAssistant();

  const rootRef = useRef<HTMLDivElement>(null);
  const dragState = useRef({
    startX: 0, startY: 0, originX: 0, originY: 0, moved: false, pointerId: -1,
  });

  useEffect(() => {
    localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0');
  }, [collapsed]);

  const getDefaultPos = useCallback(() => ({
    x: window.innerWidth - PANEL_WIDTH - 24,
    y: window.innerHeight - PANEL_HEIGHT - 24,
  }), []);

  const toggle = () => setCollapsed((v) => !v);

  // 使用 Pointer Events + setPointerCapture：即使鼠标飞出窗口也能收到事件
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.preventDefault();
    const root = rootRef.current;
    if (!root) return;
    const el = e.currentTarget as HTMLElement;

    // 关键：捕获指针，所有后续事件都发到这个元素，即使鼠标离开窗口
    try {
      el.setPointerCapture(e.pointerId);
    } catch {
      // 某些浏览器不支持，忽略
    }

    const ds = dragState.current;
    ds.startX = e.clientX;
    ds.startY = e.clientY;
    const rect = root.getBoundingClientRect();
    ds.originX = rect.left;
    ds.originY = rect.top;
    ds.moved = false;
    ds.pointerId = e.pointerId;

    document.body.style.userSelect = 'none';
    document.body.style.cursor = 'move';

    const onMove = (ev: PointerEvent) => {
      if (ev.pointerId !== dragState.current.pointerId) return;
      const dx = ev.clientX - dragState.current.startX;
      const dy = ev.clientY - dragState.current.startY;
      if (!dragState.current.moved && (Math.abs(dx) > 3 || Math.abs(dy) > 3)) {
        dragState.current.moved = true;
      }
      if (!dragState.current.moved) return;
      const w = collapsed ? BUBBLE_SIZE : PANEL_WIDTH;
      const h = collapsed ? BUBBLE_SIZE : PANEL_HEIGHT;
      const maxX = window.innerWidth - w - 8;
      const maxY = window.innerHeight - h - 8;
      const x = Math.min(Math.max(8, dragState.current.originX + dx), Math.max(8, maxX));
      const y = Math.min(Math.max(8, dragState.current.originY + dy), Math.max(8, maxY));
      root.style.left = `${x}px`;
      root.style.top = `${y}px`;
    };

    const onUp = (ev: PointerEvent) => {
      if (ev.pointerId !== dragState.current.pointerId) return;
      try {
        el.releasePointerCapture(ev.pointerId);
      } catch {
        // 忽略
      }
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      document.removeEventListener('pointercancel', onUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
      dragState.current.pointerId = -1;

      if (dragState.current.moved) {
        const rect = root.getBoundingClientRect();
        setPosition({ x: rect.left, y: rect.top });
      } else {
        setCollapsed((v) => !v);
      }
    };

    document.addEventListener('pointermove', onMove);
    document.addEventListener('pointerup', onUp);
    document.addEventListener('pointercancel', onUp);
  }, [collapsed]);

  const pos = position || getDefaultPos();

  return (
    <div
      ref={rootRef}
      className={`fa-root ${collapsed ? 'fa-collapsed' : 'fa-expanded'}`}
      style={{ left: pos.x, top: pos.y }}
    >
      {collapsed ? (
        <div className="fa-bubble" onPointerDown={handlePointerDown} title="平台助手">
          <img src={tubiaoImg} alt="平台助手" className="fa-bubble-icon" />
        </div>
      ) : (
        <>
          <div className="fa-drag-handle" onPointerDown={handlePointerDown} />
          <AssistantPanel
            messages={messages}
            sessions={sessions}
            currentSessionId={currentSessionId}
            loading={loading}
            initialized={initialized}
            onSend={send}
            onStop={stop}
            onClose={toggle}
            onClear={clearHistory}
            onRegenerate={regenerate}
            onSwitchSession={switchSession}
            onNewSession={newSession}
          />
        </>
      )}
    </div>
  );
};

export default FloatingAssistant;
