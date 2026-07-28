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
    data?: {
        xData: any[]
        yData: any[]
    }
    isNoData?: boolean
}

const defaultChartStyle: React.CSSProperties = {
    height: 300
}

export default function EchartCore(props: IProps) {
    const chartRef = useRef<HTMLDivElement>(null);
    const chartInstanceRef = useRef<echarts.ECharts | null>(null);
    const { t } = useTranslation();

    useEffect(() => {
        if (!chartRef.current) return;

        const chart = echarts.init(chartRef.current);
        chartInstanceRef.current = chart;
        chart.setOption(props.option);

        return () => {
            if (chartInstanceRef.current && !chartInstanceRef.current.isDisposed()) {
                chartInstanceRef.current.dispose();
            }
            chartInstanceRef.current = null;
        };
    }, [props.option]);

    return (
        <Spin spinning={!!props.loading}>
            <div className="chart-container">
                <div ref={chartRef} style={{ ...defaultChartStyle, ...props.style }}></div>
                {
                    props.isNoData ? <div className="chart-nodata">
                        <div>{t('暂无数据')}</div>
                    </div> : null
                }
            </div>
        </Spin>
    )
}
