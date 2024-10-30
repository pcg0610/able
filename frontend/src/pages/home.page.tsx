import { Link } from 'react-router-dom';

const HomePage = () => {
  return (
    <>
      <div>홈화면</div>
      <Link to={'/canvas'}>캔버스</Link>
      &nbsp;
      <Link to={'/train'}>학습</Link>
    </>
  );
};

export default HomePage;
