import React, { useEffect, useRef } from 'react'
import * as echarts from 'echarts';
import './EchartCore.less';
import { Spin } from 'antd';
import { useTranslation } from 'react-i18next';

export type ECOption = echarts.EChartsOption

interface IProps {
    option: echarts.EChartsOption
    loading?: boolean
    title?: string
    style?: React.CSSProperties
    unit?: string
    data?: { xData: any[]; yData: any[] }
    isNoData?: boolean
    /** 推理监控专用：保留用户图例选择状态 */
    preserveLegendSelection?: boolean
    /** 当此 key 变化时清空旧图例选择（用于服务切换） */
    legendStateKey?: string
}

/** 从 ECharts option 中安全提取 legend.selected */
function getLegendSelected(option: any): Record<string, boolean> | null {
    try {
        const legend = option?.legend;
        if (!legend) return null;
        if (Array.isArray(legend) && legend.length > 0 && legend[0]?.selected) {
            return { ...legend[0].selected };
        }
        if (legend.selected) return { ...legend.selected };
    } catch { /* */ }
    return null;
}

/** 将 legend.selected 合并到 option 中，返回新 option（不修改原对象） */
function mergeLegendSelected(option: any, selected: Record<string, boolean> | null): any {
    if (!selected || Object.keys(selected).length === 0) return option;

    const seriesNames = new Set(
        (option.series || []).map((s: any) => s.name).filter(Boolean),
    );
    const valid: Record<string, boolean> = {};
    for (const [name, visible] of Object.entries(selected)) {
        if (seriesNames.has(name)) valid[name] = visible;
    }
    if (Object.keys(valid).length === 0) return option;

    const legends = Array.isArray(option.legend) ? [...option.legend] : [{ ...(option.legend || {}) }];
    legends[0] = { ...legends[0], selected: { ...(legends[0]?.selected || {}), ...valid } };

    return { ...option, legend: legends };
}

export default function EchartCore(props: IProps) {
    const rootRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<HTMLDivElement>(null);
    const chartInstanceRef = useRef<echarts.ECharts | null>(null);
    const frameRef = useRef<number | null>(null);
    const optionRef = useRef(props.option);
    const warnOnceRef = useRef(false);
    const selectedRef = useRef<Record<string, boolean> | null>(null);
    const lastLegendKeyRef = useRef<string | undefined>(undefined);
    const { t } = useTranslation();

    // 始终保持最新 option 引用
    useEffect(() => { optionRef.current = props.option; }, [props.option]);

    // legendStateKey 变化时清空缓存的选择
    if (props.legendStateKey !== lastLegendKeyRef.current) {
        selectedRef.current = null;
        lastLegendKeyRef.current = props.legendStateKey;
    }

    const initChart = () => {
        const el = chartRef.current;
        if (!el) return;
        const w = el.clientWidth;
        const h = el.clientHeight;
        if (w <= 0 || h <= 0) {
            // 容器尺寸为零时不能初始化 ECharts，等待 ResizeObserver 触发非零尺寸
            if (!warnOnceRef.current) console.warn('[EchartCore] zero size — waiting', { w, h });
            warnOnceRef.current = true;
            return;
        }
        // 尺寸变为有效，重置 warn flag 以便未来再次出现时能记录
        warnOnceRef.current = false;
        if (!chartInstanceRef.current || chartInstanceRef.current.isDisposed()) {
            chartInstanceRef.current = echarts.init(el);
        }
        // 初始化或 resize 后始终用最新 option 渲染
        chartInstanceRef.current.setOption(optionRef.current, { notMerge: true });
    };

    useEffect(() => {
        const root = rootRef.current;
        if (!root) return;
        const initTimer = setTimeout(() => initChart(), 50);
        const ro = new ResizeObserver(() => {
            if (frameRef.current !== null) cancelAnimationFrame(frameRef.current);
            frameRef.current = requestAnimationFrame(() => {
                const inst = chartInstanceRef.current;
                if (inst && !inst.isDisposed()) { inst.resize(); } else { initChart(); }
                frameRef.current = null;
            });
        });
        ro.observe(root);
        return () => {
            clearTimeout(initTimer);
            ro.disconnect();
            if (frameRef.current !== null) { cancelAnimationFrame(frameRef.current); frameRef.current = null; }
            if (chartInstanceRef.current && !chartInstanceRef.current.isDisposed()) {
                chartInstanceRef.current.dispose();
            }
            chartInstanceRef.current = null;
        };
    }, []);

    // option 更新：单次原子 setOption，将 legend.selected 合并进完整 option
    useEffect(() => {
        const inst = chartInstanceRef.current;
        if (inst && !inst.isDisposed()) {
            let finalOption = props.option;

            if (props.preserveLegendSelection) {
                // 服务切换（legendStateKey 变化）时不保留旧选择
                const currentKey = props.legendStateKey;
                if (currentKey !== lastLegendKeyRef.current) {
                    selectedRef.current = null;
                    lastLegendKeyRef.current = currentKey;
                }
                // 首次：从当前实例读取已选择的状态
                if (!selectedRef.current) {
                    selectedRef.current = getLegendSelected(inst.getOption());
                }
                finalOption = mergeLegendSelected(finalOption, selectedRef.current);
            }

            // 只调用一次 setOption，完整的 option
            inst.setOption(finalOption, { notMerge: true });

            // 更新后读取最新 legend.selected 保存
            if (props.preserveLegendSelection) {
                selectedRef.current = getLegendSelected(inst.getOption());
            }
        } else {
            initChart();
        }
    }, [props.option]);

    return (
        <div ref={rootRef} className="echart-core-root" style={props.style}>
            <div ref={chartRef} className="echart-core-canvas" />
            {props.loading && (
                <div className="echart-core-overlay"><Spin spinning /></div>
            )}
            {props.isNoData && (
                <div className="echart-core-overlay echart-core-nodata">
                    <span>{t('暂无数据')}</span>
                </div>
            )}
        </div>
    )
}
