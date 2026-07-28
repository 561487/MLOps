import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Row, Col, Spin, Empty } from 'antd';
import TitleHeader from '../../components/TitleHeader/TitleHeader';
import FilterBar from './components/FilterBar';
import MetricCards from './components/MetricCards';
import TrendCharts from './components/TrendCharts';
import ServiceList from './components/ServiceList';
import { fetchServices, fetchSummary } from './api';
import { getMonitorDisplayStatus, DISPLAY_STATUS_CONFIG } from './status';
import type { ServiceItem, SummaryResult, TimeRange, StatusFilter } from './types';
import './InferenceMonitor.less';

const SUPPORTED_ENGINES = ['vllm'];

interface Props { breadcrumbs?: string[]; }

/** 判断当前选中服务是否可以请求监控数据 */
function isMonitorEnabled(service: ServiceItem | null): boolean {
  if (!service) return false;
  return service.model_status === 'online'
    && SUPPORTED_ENGINES.includes(String(service.service_type || '').toLowerCase());
}

/** 获取监控禁用原因 */
function getDisabledReason(service: ServiceItem | null): string | undefined {
  if (!service) return undefined;
  if (service.model_status !== 'online') return 'service_stopped';
  return undefined;
}

const InferenceMonitor: React.FC<Props> = ({ breadcrumbs }) => {
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [services, setServices] = useState<ServiceItem[]>([]);
  const [selectedService, setSelectedService] = useState<ServiceItem | null>(null);
  const [summary, setSummary] = useState<SummaryResult | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  // 刷新令牌：每次手动或自动刷新时递增，通知 TrendCharts 重新请求 timeseries
  const [refreshToken, setRefreshToken] = useState(0);
  // 标记本次刷新是否为手动（手动刷新绕过缓存）
  const [forceRefresh, setForceRefresh] = useState(false);

  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [timeRange, setTimeRange] = useState<TimeRange>('6h');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const refreshTimerRef = useRef<number | null>(null);
  const selectedServiceIdRef = useRef<number | null>(null);
  // 用 ref 避免定时器闭包持有旧的 selectedService
  const selectedServiceRef = useRef<ServiceItem | null>(null);

  // 切换服务时清空上一个服务的 summary 和错误
  useEffect(() => {
    setSummary(null);
    setSummaryError(null);
  }, [selectedService?.service_id]);

  const loadServices = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchServices({
        engine: 'vllm',
        status: statusFilter === 'all' ? undefined : statusFilter,
      });
      setServices(data.filter((s: ServiceItem) =>
        SUPPORTED_ENGINES.includes(String(s.service_type || '').toLowerCase())
      ));
    } catch { /* silent */ } finally { setLoading(false); }
  }, [statusFilter]);

  const loadSummaryFn = useCallback(async (serviceId: number, force?: boolean) => {
    setSummaryLoading(true);
    setSummaryError(null);

    try {
      const result = await fetchSummary(serviceId, { force });
      if (serviceId !== selectedServiceIdRef.current) return;
      setSummary(result);
    } catch {
      if (serviceId !== selectedServiceIdRef.current) return;
      setSummary(null);
      setSummaryError('无法获取监控摘要');
    } finally {
      if (serviceId === selectedServiceIdRef.current) {
        setSummaryLoading(false);
      }
    }
  }, []);

  // ====== 统一刷新函数 ======
  const refreshAll = useCallback(async (options?: { force?: boolean }) => {
    const serviceId = selectedServiceIdRef.current;
    const service = selectedServiceRef.current;
    const force = options?.force === true;

    setRefreshing(true);

    try {
      await loadServices();

      if (serviceId && service && isMonitorEnabled(service)) {
        await loadSummaryFn(serviceId, force);
        // 递增 refreshToken 触发 TrendCharts 重新加载
        setForceRefresh(force);
        setRefreshToken(current => current + 1);
      }
    } finally {
      setRefreshing(false);
    }
  }, [loadServices, loadSummaryFn]);

  useEffect(() => { loadServices(); }, [loadServices]);

  const handleSelect = useCallback((s: ServiceItem) => {
    setSelectedService(s);
    selectedServiceRef.current = s;
    selectedServiceIdRef.current = s.service_id;
    setSummary(null);
    setSummaryError(null);

    if (isMonitorEnabled(s)) {
      loadSummaryFn(s.service_id, false);
    } else {
      setSummaryLoading(false);
    }
  }, [loadSummaryFn]);

  // ====== 自动刷新 ======
  useEffect(() => {
    if (refreshTimerRef.current) {
      clearInterval(refreshTimerRef.current);
      refreshTimerRef.current = null;
    }

    if (!autoRefresh) return;

    const service = selectedServiceRef.current;
    if (!service || !isMonitorEnabled(service)) return;

    refreshTimerRef.current = window.setInterval(() => {
      refreshAll({ force: false });
    }, 30000);

    return () => {
      if (refreshTimerRef.current) {
        clearInterval(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
    };
  }, [autoRefresh, refreshAll]);

  // ====== 手动刷新 ======
  const handleRefresh = useCallback(() => {
    refreshAll({ force: true });
  }, [refreshAll]);

  const vllmCount = services.length;
  const monitorEnabled = isMonitorEnabled(selectedService);
  const disabledReason = getDisabledReason(selectedService);

  return (
    <div className="fade-in h100 d-f fd-c inference-monitor">
      <TitleHeader
        title="推理监控"
        breadcrumbs={(breadcrumbs || ['服务化', '模型服务', '推理监控']).map((c, i) => (
          <span key={i} className="c-icon-b fs12">/<span className="plr2">{c}</span></span>
        ))}
      />
      <div className="mlr16 mb16 flex1" style={{ overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <FilterBar
          statusFilter={statusFilter} onStatusChange={setStatusFilter}
          timeRange={timeRange} onTimeRangeChange={setTimeRange}
          autoRefresh={autoRefresh} onAutoRefreshChange={setAutoRefresh}
          onRefresh={handleRefresh} loading={loading || refreshing}
        />
        <Row gutter={16} style={{ marginTop: 12, flex: 1, minHeight: 0 }} wrap={false}>
          <Col flex="0 0 400px" style={{ minWidth: 360, maxWidth: 460, overflow: 'auto' }}>
            <ServiceList
              services={services} loading={loading}
              selectedServiceId={selectedService?.service_id || null}
              onSelect={handleSelect}
            />
          </Col>
          <Col flex="1 1 0" style={{ minWidth: 0, overflow: 'auto' }}>
            {!selectedService ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 300, background: '#fff', borderRadius: 4 }}>
                <Empty description={vllmCount === 0 ? '当前没有可监控的 vLLM 推理服务' : '请选择左侧 vLLM 推理服务查看监控指标'} />
              </div>
            ) : (
              <>
                {/* 头部信息栏 */}
                <div style={{ marginBottom: 12, background: '#fff', borderRadius: 4, padding: '8px 16px' }}>
                  <span style={{ fontWeight: 600, fontSize: 15 }}>{selectedService.label || selectedService.service_name}</span>
                  <span style={{ color: '#999', marginLeft: 12, fontSize: 13 }}>
                    {selectedService.model_name ? <span>模型：{selectedService.model_name} · </span> : ''}
                    服务：{selectedService.service_name} · 引擎：{selectedService.service_type}
                    {(() => {
                      const displayStatus = getMonitorDisplayStatus(
                        selectedService,
                        summary?.monitor_status,
                      );
                      const cfg = DISPLAY_STATUS_CONFIG[displayStatus];
                      return (
                        <span style={{ color: cfg.color }}> · {cfg.label}</span>
                      );
                    })()}
                  </span>
                </div>

                {/* 已停止服务：显示空状态 */}
                {!monitorEnabled ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 300, background: '#fff', borderRadius: 4 }}>
                    <Empty description={disabledReason === 'service_stopped' ? '服务已停止，暂无监控数据' : '该服务尚未启用推理监控'} />
                  </div>
                ) : (
                  <Spin spinning={summaryLoading}>
                    {/* summary 查询失败 */}
                    {summaryError && !summary ? (
                      <div style={{ marginBottom: 12, background: '#fff', borderRadius: 4, padding: 24 }}>
                        <Empty description={summaryError} />
                      </div>
                    ) : (
                      <MetricCards summary={summary} />
                    )}
                    <TrendCharts
                      key={`${selectedService.service_id}-${selectedService.model_status}`}
                      serviceId={selectedService.service_id}
                      timeRange={timeRange}
                      enabled={monitorEnabled}
                      disabledReason={disabledReason}
                      refreshToken={refreshToken}
                      forceRefresh={forceRefresh}
                    />
                  </Spin>
                )}
              </>
            )}
          </Col>
        </Row>
      </div>
    </div>
  );
};

export default InferenceMonitor;
