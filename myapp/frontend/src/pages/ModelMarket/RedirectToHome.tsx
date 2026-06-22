// 重定向：模型市场父级菜单 → 全部模型首页

import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const RedirectToHome: React.FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    navigate('/service/model_market_group/model_market_vision', { replace: true });
  }, [navigate]);

  return null;
};

export default RedirectToHome;
