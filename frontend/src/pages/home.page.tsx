import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <>
      <div>홈화면</div>
      <Link to={'/canvas'}>캔버스</Link>
    </>
  );
};

export default Home;
