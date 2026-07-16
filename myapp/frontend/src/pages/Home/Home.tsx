import React from "react";
import { Typography } from "antd";
import { FileTextOutlined, DeploymentUnitOutlined } from "@ant-design/icons";
import FeatureCard from "./components/FeatureCard";
import PipelineList from "./components/PipelineList";
import "./Home.less";
import { useTranslation } from "react-i18next";

const { Title } = Typography;

const Home: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="home-container">
      <div className="home-content">
        {/* 平台主要功能区 */}
        <section className="home-section">
          <div className="section-header">
            <Title level={5}>
              <DeploymentUnitOutlined /> {t('平台主要功能')}
            </Title>
          </div>
          <FeatureCard />
        </section>

        {/* 流水线列表区 */}
        <section className="home-section">
          <div className="section-header">
            <Title level={5}>
              <FileTextOutlined /> {t('流水线')}
            </Title>
          </div>
          <PipelineList />
        </section>
      </div>
    </div>
  );
};

export default Home;
